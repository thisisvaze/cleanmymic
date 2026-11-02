// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "CleanMyMic",
    platforms: [.macOS(.v15)],
    targets: [
        .target(name: "ObjCTry"),
        .executableTarget(name: "CleanMyMic", dependencies: ["ObjCTry"],
                          swiftSettings: [.defaultIsolation(MainActor.self)]),
        .testTarget(name: "CleanMyMicTests", dependencies: ["CleanMyMic"]),
    ]
)
