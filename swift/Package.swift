// swift-tools-version:5.9
// DMD3C MLX-Swift Package

import PackageDescription

let package = Package(
    name: "DMD3C",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "DMD3C",
            targets: ["DMD3C"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
    ],
    targets: [
        .target(
            name: "DMD3C",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/DMD3C",
            resources: [
                .process("Resources")
            ]
        ),
    ]
)
