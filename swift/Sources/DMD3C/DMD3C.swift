import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Custom Ops (Pure MLX replacements for CUDA BpOps)

/// K-nearest neighbor search for sparse depth points.
/// Finds the 4 nearest valid sparse depth pixels for each position in the grid.
///
/// - Parameters:
///   - sparseDepth: `[B, 1, H, W]` sparse depth map (NCHW)
///   - height: spatial height
///   - width: spatial width
/// - Returns: `(offsets: [B, 2, 4, N], indices: [B, 4, N])` where N = H*W
func bpDistMLX(sparseDepth: MLXArray, height: Int, width: Int) -> (MLXArray, MLXArray) {
    let B = sparseDepth.dim(0)
    let N = height * width
    let num = 4

    let sFlat = sparseDepth.reshaped(B, 1, N)
    let valid = sFlat .> 1e-3 // [B, 1, N]

    // Create coordinate grids
    let wCoords = MLXArray(0 ..< width).asType(.float32)
    let hCoords = MLXArray(0 ..< height).asType(.float32)
    // meshgrid: xx[h,w] = w, yy[h,w] = h
    let xx = broadcast(wCoords.reshaped(1, width), to: [height, width]).reshaped(N)
    let yy = broadcast(hCoords.reshaped(height, 1), to: [height, width]).reshaped(N)

    var allArgs = [MLXArray]()
    var allOffsets = [MLXArray]()

    for b in 0 ..< B {
        let validMask = valid[b, 0] // [N]
        let validIndices = which(validMask) // [M_valid]
        let mValid = validIndices.dim(0)

        if mValid == 0 {
            allArgs.append(MLXArray.zeros([num, N], dtype: .int32))
            allOffsets.append(MLXArray.zeros([2, num, N]))
            continue
        }

        // Valid point coordinates
        let validX = xx[validIndices] // [M_valid]
        let validY = yy[validIndices] // [M_valid]

        // Distance from each pixel to each valid point
        let dx = xx.reshaped(N, 1) - validX.reshaped(1, mValid) // [N, M_valid]
        let dy = yy.reshaped(N, 1) - validY.reshaped(1, mValid) // [N, M_valid]
        let dists = dx * dx + dy * dy // [N, M_valid]

        // Find k nearest
        let k = min(num, mValid)
        let sortedIdx = argSort(dists, axis: 1) // [N, M_valid]
        var topkIdx: MLXArray
        if mValid < num {
            let lastCol = sortedIdx[0..., (mValid - 1) ..< mValid] // [N, 1]
            let pad = broadcast(lastCol, to: [N, num - mValid])
            topkIdx = concatenated([sortedIdx, pad], axis: 1)[0..., 0 ..< num]
        } else {
            topkIdx = sortedIdx[0..., 0 ..< num] // [N, num]
        }

        // Map to global indices
        let topkGlobal = validIndices[topkIdx] // [N, num]

        // Compute offsets
        let neighborX = validX[topkIdx] // [N, num]
        let neighborY = validY[topkIdx] // [N, num]
        let offX = neighborX - xx.reshaped(N, 1) // [N, num]
        let offY = neighborY - yy.reshaped(N, 1) // [N, num]

        allArgs.append(topkGlobal.transposed(1, 0)) // [num, N]
        allOffsets.append(stacked([offX.transposed(), offY.transposed()])) // [2, num, N]
    }

    let args = stacked(allArgs) // [B, num, N]
    let offsets = stacked(allOffsets) // [B, 2, num, N]
    return (offsets, args)
}

/// Position-dependent (spatially-varying) convolution.
/// Each spatial position has its own KxK convolution kernel.
///
/// - Parameters:
///   - x: `[B, C, H, W]` input features (NCHW)
///   - weight: `[B, C*K*K, H, W]` per-position kernels
/// - Returns: `[B, C, H, W]` convolved output
func conv2dLocal(_ x: MLXArray, weight: MLXArray) -> MLXArray {
    let B = x.dim(0)
    let C = x.dim(1)
    let H = x.dim(2)
    let W = x.dim(3)
    let K2 = weight.dim(1) / C
    let K = Int(sqrt(Double(K2)))
    let pad = (K - 1) / 2

    // Pad input
    let xPad = padded(x, widths: [.init(0, 0), .init(0, 0), .init(pad, pad), .init(pad, pad)])

    // Extract patches and multiply with weights
    var patches = [MLXArray]()
    for i in 0 ..< K {
        for j in 0 ..< K {
            patches.append(xPad[0..., 0..., i ..< (i + H), j ..< (j + W)])
        }
    }
    let patchStack = stacked(patches, axis: 2) // [B, C, K*K, H, W]
    let wReshaped = weight.reshaped(B, C, K2, H, W)
    let output = sum(patchStack * wReshaped, axis: 2) // [B, C, H, W]
    return output
}

// MARK: - Network Building Blocks

/// Conv2d + BatchNorm + Activation
class Conv2dBnAct: Module {
    let conv: Conv2d
    let bn: BatchNorm?
    let activation: String

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 3,
         stride: Int = 1, padding: Int = 1, bias: Bool = false,
         useBn: Bool = true, activation: String = "relu") {
        self.conv = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: .init(kernelSize, kernelSize),
            stride: .init(stride, stride),
            padding: .init(padding, padding),
            bias: !useBn
        )
        self.bn = useBn ? BatchNorm(featureCount: outChannels) : nil
        self.activation = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        if let bn = bn { out = bn(out) }
        switch activation {
        case "relu": return relu(out)
        case "gelu": return gelu(out)
        case "sigmoid": return sigmoid(out)
        default: return out
        }
    }
}

/// ConvTranspose2d + BatchNorm + ReLU
class ConvTranspose2dBnRelu: Module {
    let conv: ConvTranspose2d
    let bn: BatchNorm?

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 4,
         stride: Int = 2, padding: Int = 1, useBn: Bool = true) {
        self.conv = ConvTranspose2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: .init(kernelSize, kernelSize),
            stride: .init(stride, stride),
            padding: .init(padding, padding),
            bias: !useBn
        )
        self.bn = useBn ? BatchNorm(featureCount: outChannels) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        if let bn = bn { out = bn(out) }
        return relu(out)
    }
}

/// Downsample block (1x1 conv + BN)
class DownsampleBlock: Module {
    let conv: Conv2d
    let bn: BatchNorm

    init(inChannels: Int, outChannels: Int, stride: Int) {
        self.conv = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: .init(1, 1), stride: .init(stride, stride), bias: false
        )
        self.bn = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return bn(conv(x))
    }
}

/// ResNet BasicBlock
class ResBasicBlock: Module {
    let conv1: Conv2d
    let bn1: BatchNorm
    let conv2: Conv2d
    let bn2: BatchNorm
    let downsample: DownsampleBlock?

    init(inplanes: Int, planes: Int, stride: Int = 1, downsample: DownsampleBlock? = nil) {
        self.conv1 = Conv2d(
            inputChannels: inplanes, outputChannels: planes,
            kernelSize: .init(3, 3), stride: .init(stride, stride),
            padding: .init(1, 1), bias: false
        )
        self.bn1 = BatchNorm(featureCount: planes)
        self.conv2 = Conv2d(
            inputChannels: planes, outputChannels: planes,
            kernelSize: .init(3, 3), stride: .init(1, 1),
            padding: .init(1, 1), bias: false
        )
        self.bn2 = BatchNorm(featureCount: planes)
        self.downsample = downsample
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var identity = x
        var out = relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        if let ds = downsample { identity = ds(x) }
        return relu(out + identity)
    }
}

/// Generates spatially-varying convolution kernels for CSPN
class GenKernel: Module {
    let conv0: Conv2dBnAct
    let conv1: Conv2dBnAct
    let eps: Float

    init(inChannels: Int, pk: Int, eps: Float = 1e-6) {
        self.eps = eps
        self.conv0 = Conv2dBnAct(inChannels: inChannels, outChannels: inChannels)
        self.conv1 = Conv2dBnAct(inChannels: inChannels, outChannels: pk * pk - 1, activation: "identity")
    }

    func callAsFunction(_ fout: MLXArray) -> MLXArray {
        var weight = conv1(conv0(fout))
        let weightSum = sum(abs(weight), axis: -1, keepDims: true)
        weight = weight / (weightSum + eps)
        let weightMid = 1.0 - sum(weight, axis: -1, keepDims: true)
        let half = weight.dim(-1) / 2
        let weightPre = weight[0..., 0..., 0..., 0 ..< half]
        let weightPost = weight[0..., 0..., 0..., half...]
        return concatenated([weightPre, weightMid, weightPost], axis: -1)
    }
}

/// Coefficient prediction module
class CoefModule: Module {
    let conv: Conv2d

    init(inChannels: Int) {
        self.conv = Conv2d(inputChannels: inChannels, outputChannels: 3,
                           kernelSize: .init(1, 1), bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let feat = conv(x)
        return (feat[0..., 0..., 0..., 0 ..< 1],
                feat[0..., 0..., 0..., 1 ..< 2],
                feat[0..., 0..., 0..., 2 ..< 3])
    }
}

/// Depth-to-space pixel shuffle upsampling with learned weights
class PermuteModule: Module {
    let stride: Int
    let outCh: Int
    let conv0: Conv2dBnAct
    let conv1: Conv2dBnAct
    let conv2: Conv2d

    init(inChannels: Int, outChannels: Int = 1, stride: Int = 2) {
        self.stride = stride
        self.outCh = outChannels
        self.conv0 = Conv2dBnAct(inChannels: inChannels, outChannels: inChannels, kernelSize: 1, padding: 0)
        self.conv1 = Conv2dBnAct(inChannels: inChannels, outChannels: inChannels, kernelSize: 1, padding: 0)
        self.conv2 = Conv2d(inputChannels: inChannels, outputChannels: outChannels * stride * stride,
                            kernelSize: .init(1, 1), bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv2(conv1(conv0(x))) // [B, H, W, c*s*s]
        let B = out.dim(0), H = out.dim(1), W = out.dim(2)
        let s = stride, c = outCh
        out = out.reshaped(B, H, W, c, s, s)
        out = out.transposed(0, 1, 4, 2, 5, 3) // [B, H, s, W, s, c]
        return out.reshaped(B, H * s, W * s, c)
    }
}

/// Weighted pooling of sparse depth at a given pyramid level
class WPoolModule: Module {
    let level: Int
    let drift: Float
    let permuteModule: PermuteModule

    init(inChannels: Int, level: Int, drift: Float = 1e6) {
        self.level = level
        self.drift = drift
        let stride = 1 << level // 2^level
        self.permuteModule = PermuteModule(inChannels: inChannels, outChannels: 1, stride: stride)
    }

    func callAsFunction(S: MLXArray, fout: MLXArray) -> MLXArray {
        let W = permuteModule(fout) // [B, H, W, 1]
        let size = 1 << level
        let M = (S .> 1e-3).asType(.float32)

        // Convert to NCHW for pooling
        let wNchw = W.transposed(0, 3, 1, 2) // [B, 1, H, W]
        let mNchw = M.transposed(0, 3, 1, 2)
        let sNchw = S.transposed(0, 3, 1, 2)

        let B = wNchw.dim(0), C = wNchw.dim(1), Hh = wNchw.dim(2), Ww = wNchw.dim(3)
        let Hp = Hh / size, Wp = Ww / size

        // Max pool via reshape
        let wm = (wNchw + drift) * mNchw
        let wmBlocks = wm.reshaped(B, C, Hp, size, Wp, size)
        var maxW = wmBlocks.max(axes: [3, 5]) // [B, C, Hp, Wp]

        // Nearest-neighbor upsample
        maxW = repeated(repeated(maxW, count: size, axis: 2), count: size, axis: 3)
        maxW = (maxW - drift) * mNchw

        let expW = exp(wNchw * mNchw - maxW) * mNchw

        // Average pool
        let sExpW = sNchw * expW
        let sExpWBlocks = sExpW.reshaped(B, C, Hp, size, Wp, size)
        let avgS = sExpWBlocks.sum(axes: [3, 5]) / Float(size * size)

        let expWBlocks = expW.reshaped(B, C, Hp, size, Wp, size)
        let avgExpW = expWBlocks.sum(axes: [3, 5]) / Float(size * size)

        let sp = avgS / (avgExpW + 1e-6)
        return sp.transposed(0, 2, 3, 1) // back to NHWC
    }
}

/// Upsample + Concatenate + Conv for PMP inter-level connections
class UpCatModule: Module {
    let upf: ConvTranspose2dBnRelu
    let convBlock: Conv2dBnAct

    init(inChannels: Int, outChannels: Int) {
        self.upf = ConvTranspose2dBnRelu(inChannels: inChannels + 1, outChannels: outChannels)
        self.convBlock = Conv2dBnAct(inChannels: outChannels * 2, outChannels: outChannels)
    }

    func callAsFunction(y: MLXArray, x: MLXArray, d: MLXArray) -> MLXArray {
        let upIn = concatenated([x, d], axis: -1)
        let upOut = upf(upIn)
        let catOut = concatenated([upOut, y], axis: -1)
        return convBlock(catOut)
    }
}

/// Upsample + Concatenate + Conv for UBNet decoder
class UpCCModule: Module {
    let upf: ConvTranspose2dBnRelu
    let convBlock: Conv2dBnAct

    init(inChannels: Int, midChannels: Int, outChannels: Int) {
        self.upf = ConvTranspose2dBnRelu(inChannels: inChannels, outChannels: outChannels)
        self.convBlock = Conv2dBnAct(inChannels: midChannels + outChannels, outChannels: outChannels)
    }

    func callAsFunction(x: MLXArray, y: MLXArray) -> MLXArray {
        let upOut = upf(x)
        let catOut = concatenated([upOut, y], axis: -1)
        return convBlock(catOut)
    }
}

/// Propagation module - propagates depth from sparse neighbors
class PropModule: Module {
    let convXF0: Conv2dBnAct
    let convXF1: Conv2dBnAct
    let convXL0: Conv2dBnAct
    let convXL1: Conv2dBnAct
    let coef: CoefModule

    init(cfi: Int, cfp: Int = 3, cfo: Int = 2) {
        let ct = cfo + cfi + cfi + cfp
        self.convXF0 = Conv2dBnAct(inChannels: ct, outChannels: cfi, kernelSize: 1, padding: 0, activation: "gelu")
        self.convXF1 = Conv2dBnAct(inChannels: cfi, outChannels: cfi, kernelSize: 1, padding: 0, activation: "gelu")
        self.convXL0 = Conv2dBnAct(inChannels: cfi, outChannels: cfi, kernelSize: 1, padding: 0, activation: "gelu")
        self.convXL1 = Conv2dBnAct(inChannels: cfi, outChannels: cfi, kernelSize: 1, padding: 0, activation: "identity")
        self.coef = CoefModule(inChannels: cfi)
    }

    func callAsFunction(ifNhwc: MLXArray, pf: MLXArray, ofnum: MLXArray, args: MLXArray) -> MLXArray {
        let num = args.dim(1)
        let B = ifNhwc.dim(0), H = ifNhwc.dim(1), W = ifNhwc.dim(2), cfi = ifNhwc.dim(3)
        let N = H * W
        let cfp = pf.dim(1)

        // Convert features to flat NCHW
        let ifFlat = ifNhwc.transposed(0, 3, 1, 2).reshaped(B, cfi, 1, N) // [B, Cfi, 1, N]

        // Gather neighbor features using args indices
        // args: [B, num, N] - indices into N
        var ipfnumParts = [MLXArray]()
        var pfnumParts = [MLXArray]()

        for b in 0 ..< B {
            let ifB = ifFlat[b] // [Cfi, 1, N]
            let pfB = pf[b] // [Cfp, M]
            let argsB = args[b] // [num, N]

            for c in 0 ..< cfi {
                let plane = ifB[c, 0] // [N]
                ipfnumParts.append(plane[argsB]) // [num, N]
            }
            for c in 0 ..< cfp {
                let plane = pfB[c] // [M]
                pfnumParts.append(plane[argsB]) // [num, N]
            }
        }

        let ipfnum = stacked(ipfnumParts).reshaped(B, cfi, num, N)
        let pfnum = stacked(pfnumParts).reshaped(B, cfp, num, N)
        let ifnum = broadcast(ifFlat, to: [B, cfi, num, N])

        // Concatenate all features
        let x = concatenated([ifnum, ipfnum, pfnum, ofnum], axis: 1) // [B, Ct, num, N]

        // Process through 1x1 convolutions (reshape for NHWC processing)
        let xNhwc = x.transposed(0, 2, 3, 1).reshaped(B * num, 1, N, -1)

        var xf = convXF1(convXF0(xNhwc))
        let xl = convXL1(convXL0(xf))
        xf = gelu(xf + xl)

        let (alpha, beta, omega) = coef(xf)

        // Reshape back
        let alphaR = alpha.reshaped(B, num, N, 1).transposed(0, 3, 1, 2)
        let betaR = beta.reshaped(B, num, N, 1).transposed(0, 3, 1, 2)
        var omegaR = omega.reshaped(B, num, N, 1).transposed(0, 3, 1, 2)
        omegaR = softmax(omegaR, axis: 2)

        let depthNeighbors = pfnum[0..., (cfp - 1)..., 0..., 0...].reshaped(B, 1, num, N)

        let dout = sum(((alphaR + 1) * depthNeighbors + betaR) * omegaR, axis: 2, keepDims: true)
        return dout.reshaped(B, 1, H, W).transposed(0, 2, 3, 1) // [B, H, W, 1] NHWC
    }
}

/// U-shaped fusion network
class UBNetModule: Module {
    let encoderInitConv: Conv2dBnAct?
    let encoderBlocks: [[ResBasicBlock]]
    let decoderLayers: [UpCCModule]

    init(inplanes: Int, dplanes: Int = 1, blocknum: Int = 2, depth: Int = 1) {
        let bc = inplanes / 2
        var encBlocks = [[ResBasicBlock]]()
        var decLayers = [UpCCModule]()

        // First encoder layer
        self.encoderInitConv = Conv2dBnAct(inChannels: inplanes + dplanes, outChannels: bc * 2)
        var firstBlocks = [ResBasicBlock]()
        for _ in 0 ..< blocknum {
            firstBlocks.append(ResBasicBlock(inplanes: bc * 2, planes: bc * 2))
        }
        encBlocks.append(firstBlocks)

        // Additional encoder layers
        var inCh = bc * 2
        for _ in 0 ..< depth {
            let outCh = min(inCh * 2, 256)
            var blocks = [ResBasicBlock]()
            let ds = DownsampleBlock(inChannels: inCh, outChannels: outCh, stride: 2)
            blocks.append(ResBasicBlock(inplanes: inCh, planes: outCh, stride: 2, downsample: ds))
            for _ in 1 ..< blocknum {
                blocks.append(ResBasicBlock(inplanes: outCh, planes: outCh))
            }
            encBlocks.append(blocks)
            decLayers.append(UpCCModule(inChannels: outCh, midChannels: inCh, outChannels: inCh))
            inCh = min(inCh * 2, 256)
        }

        self.encoderBlocks = encBlocks
        self.decoderLayers = decLayers
    }

    func callAsFunction(x: MLXArray, d: MLXArray?) -> MLXArray {
        var current = x
        if let d = d {
            current = concatenated([current, d], axis: -1)
        }

        var feat = [MLXArray]()
        for (i, blocks) in encoderBlocks.enumerated() {
            if i == 0, let initConv = encoderInitConv {
                current = initConv(current)
            }
            for block in blocks {
                current = block(current)
            }
            feat.append(current)
        }

        var out = feat.last!
        for idx in stride(from: feat.count - 2, through: 0, by: -1) {
            out = decoderLayers[idx](x: out, y: feat[idx])
        }
        return out
    }
}

/// CSPN++ implementation
class CSPNModule: Module {
    let pt: Int
    let weight3x3: GenKernel
    let weight5x5: GenKernel
    let weight7x7: GenKernel
    let convmask0: Conv2dBnAct
    let convmask1: Conv2dBnAct
    let convck0: Conv2dBnAct
    let convck1: Conv2dBnAct
    let convct0: Conv2dBnAct
    let convct1: Conv2dBnAct

    init(inChannels: Int, pt: Int) {
        self.pt = pt
        self.weight3x3 = GenKernel(inChannels: inChannels, pk: 3)
        self.weight5x5 = GenKernel(inChannels: inChannels, pk: 5)
        self.weight7x7 = GenKernel(inChannels: inChannels, pk: 7)
        self.convmask0 = Conv2dBnAct(inChannels: inChannels, outChannels: inChannels)
        self.convmask1 = Conv2dBnAct(inChannels: inChannels, outChannels: 3, useBn: false, activation: "sigmoid")
        self.convck0 = Conv2dBnAct(inChannels: inChannels, outChannels: inChannels)
        self.convck1 = Conv2dBnAct(inChannels: inChannels, outChannels: 3, useBn: false, activation: "identity")
        self.convct0 = Conv2dBnAct(inChannels: inChannels + 3, outChannels: inChannels)
        self.convct1 = Conv2dBnAct(inChannels: inChannels, outChannels: 3, useBn: false, activation: "identity")
    }

    private func applyLocalConv(hn: MLXArray, weight: MLXArray) -> MLXArray {
        // hn: [B, C, H, W] NCHW, weight: [B, H, W, K*K] NHWC
        let B = hn.dim(0), C = hn.dim(1)
        let K2 = weight.dim(-1)
        let wNchw = weight.transposed(0, 3, 1, 2) // [B, K*K, H, W]
        // Expand for all channels
        let wExpanded = repeated(expandedDimensions(wNchw, axis: 1), count: C, axis: 1)
            .reshaped(B, C * K2, hn.dim(2), hn.dim(3))
        return conv2dLocal(hn, weight: wExpanded)
    }

    func callAsFunction(fout: MLXArray, hn: MLXArray, h0: MLXArray) -> MLXArray {
        let w3 = weight3x3(fout) // [B, H, W, 9]
        let w5 = weight5x5(fout) // [B, H, W, 25]
        let w7 = weight7x7(fout) // [B, H, W, 49]

        var maskAll = convmask1(convmask0(fout)) // [B, H, W, 3]
        let h0Valid = (h0 .> 1e-3).asType(.float32)
        maskAll = maskAll * h0Valid

        let mask3 = maskAll[0..., 0..., 0..., 0 ..< 1]
        let mask5 = maskAll[0..., 0..., 0..., 1 ..< 2]
        let mask7 = maskAll[0..., 0..., 0..., 2 ..< 3]

        let confAll = softmax(convck1(convck0(fout)), axis: -1) // [B, H, W, 3]
        let conf3 = confAll[0..., 0..., 0..., 0 ..< 1]
        let conf5 = confAll[0..., 0..., 0..., 1 ..< 2]
        let conf7 = confAll[0..., 0..., 0..., 2 ..< 3]

        // Convert to NCHW
        var hn3 = hn.transposed(0, 3, 1, 2)
        var hn5 = hn.transposed(0, 3, 1, 2)
        var hn7 = hn.transposed(0, 3, 1, 2)
        let h0Nchw = h0.transposed(0, 3, 1, 2)
        let m3 = mask3.transposed(0, 3, 1, 2)
        let m5 = mask5.transposed(0, 3, 1, 2)
        let m7 = mask7.transposed(0, 3, 1, 2)

        var hns = [hn] // NHWC intermediates

        for i in 0 ..< pt {
            hn3 = (1.0 - m3) * applyLocalConv(hn: hn3, weight: w3) + m3 * h0Nchw
            hn5 = (1.0 - m5) * applyLocalConv(hn: hn5, weight: w5) + m5 * h0Nchw
            hn7 = (1.0 - m7) * applyLocalConv(hn: hn7, weight: w7) + m7 * h0Nchw

            if i == pt / 2 - 1 {
                let c3 = conf3.transposed(0, 3, 1, 2)
                let c5 = conf5.transposed(0, 3, 1, 2)
                let c7 = conf7.transposed(0, 3, 1, 2)
                let mid = c3 * hn3 + c5 * hn5 + c7 * hn7
                hns.append(mid.transposed(0, 2, 3, 1))
            }
        }

        let c3f = conf3.transposed(0, 3, 1, 2)
        let c5f = conf5.transposed(0, 3, 1, 2)
        let c7f = conf7.transposed(0, 3, 1, 2)
        let final = c3f * hn3 + c5f * hn5 + c7f * hn7
        hns.append(final.transposed(0, 2, 3, 1))

        let hnsCat = concatenated(hns, axis: -1) // [B, H, W, 3]
        let wtIn = concatenated([fout, hnsCat], axis: -1)
        let wt = softmax(convct1(convct0(wtIn)), axis: -1) // [B, H, W, 3]
        return sum(wt * hnsCat, axis: -1, keepDims: true) // [B, H, W, 1]
    }
}

// MARK: - PMP Module

/// Pre + MF + Post depth completion module at a given pyramid level
class PMPModule: Module {
    let level: Int
    let hasUp: Bool
    let hasPool: Bool

    let upcat: UpCatModule?
    let wpool: WPoolModule?
    let prop: PropModule
    let fuse: UBNetModule
    let convOut: Conv2d
    let cspn: CSPNModule

    init(level: Int, inCh: Int, outCh: Int, up: Bool = true, pool: Bool = true) {
        self.level = level
        self.hasUp = up
        self.hasPool = pool

        self.upcat = up ? UpCatModule(inChannels: inCh, outChannels: outCh) : nil
        self.wpool = pool ? WPoolModule(inChannels: outCh, level: level) : nil
        self.prop = PropModule(cfi: outCh)
        self.fuse = UBNetModule(inplanes: outCh, dplanes: 3, blocknum: 2, depth: 5 - level)
        self.convOut = Conv2d(inputChannels: outCh, outputChannels: 1,
                              kernelSize: .init(3, 3), padding: .init(1, 1), bias: true)
        self.cspn = CSPNModule(inChannels: outCh, pt: 2 * (6 - level))
    }

    func pinv(sNchw: MLXArray, K: MLXArray, xx: MLXArray, yy: MLXArray) -> MLXArray {
        let fx = K[0..., 0 ..< 1, 0 ..< 1]
        let fy = K[0..., 1 ..< 2, 1 ..< 2]
        let cx = K[0..., 0 ..< 1, 2 ..< 3]
        let cy = K[0..., 1 ..< 2, 2 ..< 3]

        let S = sNchw.reshaped(sNchw.dim(0), 1, -1)
        let xxFlat = xx.reshaped(1, 1, -1).asType(.float32)
        let yyFlat = yy.reshaped(1, 1, -1).asType(.float32)

        let px = S * (xxFlat - cx) / fx
        let py = S * (yyFlat - cy) / fy
        return concatenated([px, py, S], axis: 1) // [B, 3, N]
    }

    func callAsFunction(fout: MLXArray?, dout: MLXArray?,
                         xi: MLXArray, S: MLXArray, K: MLXArray) -> (MLXArray, MLXArray) {
        // Upsample from previous level
        var features: MLXArray
        if hasUp, let upcat = upcat, let fout = fout, let dout = dout {
            features = upcat(y: xi, x: fout, d: dout)
        } else {
            features = xi
        }

        // Pool sparse depth
        var sp: MLXArray
        if hasPool, let wpool = wpool {
            sp = wpool(S: S, fout: features)
        } else {
            sp = S
        }

        // Scale intrinsics
        let scale = Float(1 << level)
        var kp = MLXArray(K)
        // Scale fx, fy, cx, cy by dividing by 2^level
        // K: [B, 3, 3] -> scale rows 0 and 1
        let row0 = concatenated([K[0..., 0 ..< 1, 0 ..< 1] / scale,
                                  K[0..., 0 ..< 1, 1 ..< 2],
                                  K[0..., 0 ..< 1, 2 ..< 3] / scale], axis: 2)
        let row1 = concatenated([K[0..., 1 ..< 2, 0 ..< 1],
                                  K[0..., 1 ..< 2, 1 ..< 2] / scale,
                                  K[0..., 1 ..< 2, 2 ..< 3] / scale], axis: 2)
        let row2 = K[0..., 2 ..< 3, 0...]
        kp = concatenated([row0, row1, row2], axis: 1)

        let Hh = sp.dim(1), Ww = sp.dim(2)
        let wCoords = MLXArray(0 ..< Ww)
        let hCoords = MLXArray(0 ..< Hh)
        let xx = broadcast(wCoords.reshaped(1, Ww), to: [Hh, Ww])
        let yy = broadcast(hCoords.reshaped(Hh, 1), to: [Hh, Ww])

        let spNchw = sp.transposed(0, 3, 1, 2) // [B, 1, H, W]

        // === Pre: Propagation ===
        let pxyz = pinv(sNchw: spNchw, K: kp, xx: xx, yy: yy)
        let (ofnum, args) = bpDistMLX(sparseDepth: spNchw, height: Hh, width: Ww)
        var depth = prop(ifNhwc: features, pf: pxyz, ofnum: ofnum, args: args)

        // === MF: Mean Field fusion ===
        let depthNchw = depth.transposed(0, 3, 1, 2)
        let pxyz2 = pinv(sNchw: depthNchw, K: kp, xx: xx, yy: yy)
        let pxyz2Nhwc = pxyz2.reshaped(depth.dim(0), 3, Hh, Ww).transposed(0, 2, 3, 1)
        features = fuse(x: features, d: pxyz2Nhwc)
        let res = convOut(features)
        depth = depth + res

        // === Post: CSPN ===
        depth = cspn(fout: features, hn: depth, h0: sp)

        return (features, depth)
    }
}

// MARK: - Main Model

/// DMD3C depth completion model for Apple Silicon via MLX.
///
/// Inputs (all in NHWC format):
///   - image: `[B, H, W, 3]` normalized RGB image
///   - sparseDepth: `[B, H, W, 1]` sparse depth from LiDAR
///   - K: `[B, 3, 3]` camera intrinsic matrix
///
/// Output:
///   - depth: `[B, H, W, 1]` completed dense depth map
///
/// Max resolution: 500px on the long side.
public class DMD3CModel: Module {

    let bc: Int

    // Image encoder
    let convImgInit: Conv2dBnAct
    let convImgBlocks: [ResBasicBlock]
    let layer1Img: [ResBasicBlock]
    let layer2Img: [ResBasicBlock]
    let layer3Img: [ResBasicBlock]
    let layer4Img: [ResBasicBlock]
    let layer5Img: [ResBasicBlock]

    // Prediction modules (pyramid)
    let pred5: PMPModule
    let pred4: PMPModule
    let pred3: PMPModule
    let pred2: PMPModule
    let pred1: PMPModule
    let pred0: PMPModule

    public init(bc: Int = 16) {
        self.bc = bc

        self.convImgInit = Conv2dBnAct(inChannels: 3, outChannels: bc * 2)
        self.convImgBlocks = (0 ..< 2).map { _ in ResBasicBlock(inplanes: bc * 2, planes: bc * 2) }

        self.layer1Img = Self.makeEncoderLayer(inCh: bc * 2, outCh: bc * 4, numBlocks: 2, stride: 2)
        self.layer2Img = Self.makeEncoderLayer(inCh: bc * 4, outCh: bc * 8, numBlocks: 2, stride: 2)
        self.layer3Img = Self.makeEncoderLayer(inCh: bc * 8, outCh: bc * 16, numBlocks: 2, stride: 2)
        self.layer4Img = Self.makeEncoderLayer(inCh: bc * 16, outCh: bc * 16, numBlocks: 2, stride: 2)
        self.layer5Img = Self.makeEncoderLayer(inCh: bc * 16, outCh: bc * 16, numBlocks: 2, stride: 2)

        self.pred5 = PMPModule(level: 5, inCh: bc * 16, outCh: bc * 16, up: false)
        self.pred4 = PMPModule(level: 4, inCh: bc * 16, outCh: bc * 16)
        self.pred3 = PMPModule(level: 3, inCh: bc * 16, outCh: bc * 16)
        self.pred2 = PMPModule(level: 2, inCh: bc * 16, outCh: bc * 8)
        self.pred1 = PMPModule(level: 1, inCh: bc * 8, outCh: bc * 4)
        self.pred0 = PMPModule(level: 0, inCh: bc * 4, outCh: bc * 2, pool: false)
    }

    static func makeEncoderLayer(inCh: Int, outCh: Int, numBlocks: Int, stride: Int) -> [ResBasicBlock] {
        var layers = [ResBasicBlock]()
        let ds = (inCh != outCh || stride != 1)
            ? DownsampleBlock(inChannels: inCh, outChannels: outCh, stride: stride) : nil
        layers.append(ResBasicBlock(inplanes: inCh, planes: outCh, stride: stride, downsample: ds))
        for _ in 1 ..< numBlocks {
            layers.append(ResBasicBlock(inplanes: outCh, planes: outCh))
        }
        return layers
    }

    func bilinearUpsample(_ x: MLXArray, scaleFactor: Int) -> MLXArray {
        let B = x.dim(0), H = x.dim(1), W = x.dim(2), C = x.dim(3)
        let newH = H * scaleFactor, newW = W * scaleFactor

        let xNchw = x.transposed(0, 3, 1, 2) // [B, C, H, W]

        let hCoords = MLXArray(Float(0) ..< Float(newH)) * (Float(H - 1) / Float(newH - 1))
        let wCoords = MLXArray(Float(0) ..< Float(newW)) * (Float(W - 1) / Float(newW - 1))

        let h0 = floor(hCoords).asType(.int32)
        let w0 = floor(wCoords).asType(.int32)
        let h1 = minimum(h0 + 1, H - 1)
        let w1 = minimum(w0 + 1, W - 1)
        let ha = hCoords - h0.asType(.float32)
        let wa = wCoords - w0.asType(.float32)

        // Broadcast gather for bilinear
        let haB = ha.reshaped(newH, 1)
        let waB = wa.reshaped(1, newW)

        var results = [MLXArray]()
        for b in 0 ..< B {
            var channels = [MLXArray]()
            for c in 0 ..< C {
                let plane = xNchw[b, c] // [H, W]
                let v00 = plane[h0][0..., w0]
                let v01 = plane[h0][0..., w1]
                let v10 = plane[h1][0..., w0]
                let v11 = plane[h1][0..., w1]
                let out = v00 * (1 - haB) * (1 - waB) + v01 * (1 - haB) * waB +
                          v10 * haB * (1 - waB) + v11 * haB * waB
                channels.append(out)
            }
            results.append(stacked(channels)) // [C, newH, newW]
        }
        let result = stacked(results) // [B, C, newH, newW]
        return result.transposed(0, 2, 3, 1) // [B, newH, newW, C]
    }

    /// Run full forward pass.
    ///
    /// - Parameters:
    ///   - image: `[B, H, W, 3]` normalized RGB image (NHWC)
    ///   - sparseDepth: `[B, H, W, 1]` sparse depth (NHWC)
    ///   - K: `[B, 3, 3]` camera intrinsic matrix
    /// - Returns: List of 6 depth maps at different scales, each `[B, H, W, 1]`
    public func callAsFunction(image: MLXArray, sparseDepth: MLXArray, K: MLXArray) -> [MLXArray] {
        var output = [MLXArray]()

        // Encoder
        var xi0 = convImgInit(image)
        for block in convImgBlocks { xi0 = block(xi0) }

        var xi1 = xi0
        for layer in layer1Img { xi1 = layer(xi1) }

        var xi2 = xi1
        for layer in layer2Img { xi2 = layer(xi2) }

        var xi3 = xi2
        for layer in layer3Img { xi3 = layer(xi3) }

        var xi4 = xi3
        for layer in layer4Img { xi4 = layer(xi4) }

        var xi5 = xi4
        for layer in layer5Img { xi5 = layer(xi5) }

        // Decoder (coarse to fine)
        var (fout, dout) = pred5(fout: nil, dout: nil, xi: xi5, S: sparseDepth, K: K)
        output.append(bilinearUpsample(dout, scaleFactor: 32))

        (fout, dout) = pred4(fout: fout, dout: dout, xi: xi4, S: sparseDepth, K: K)
        output.append(bilinearUpsample(dout, scaleFactor: 16))

        (fout, dout) = pred3(fout: fout, dout: dout, xi: xi3, S: sparseDepth, K: K)
        output.append(bilinearUpsample(dout, scaleFactor: 8))

        (fout, dout) = pred2(fout: fout, dout: dout, xi: xi2, S: sparseDepth, K: K)
        output.append(bilinearUpsample(dout, scaleFactor: 4))

        (fout, dout) = pred1(fout: fout, dout: dout, xi: xi1, S: sparseDepth, K: K)
        output.append(bilinearUpsample(dout, scaleFactor: 2))

        (fout, dout) = pred0(fout: fout, dout: dout, xi: xi0, S: sparseDepth, K: K)
        output.append(dout)

        return output
    }

    /// Convenience: returns only the final (finest resolution) depth prediction.
    public func predict(image: MLXArray, sparseDepth: MLXArray, K: MLXArray) -> MLXArray {
        return callAsFunction(image: image, sparseDepth: sparseDepth, K: K).last!
    }
}
