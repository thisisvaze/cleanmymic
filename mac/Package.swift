// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "CleanMyMic",
    platforms: [.macOS(.v15)],
    targets: [
        .target(name: "ObjCTry"),
        .executableTarget(name: "CleanMyMic", dependencies: ["ObjCTry"],
                          swiftSettings: [.defaultIsolation(MainActor.self)],
                          // SwiftPM records the deployment target as the SDK version, and macOS then draws the app
                          // in its pre-26 look. Recording SDK 26 gives the current system look on 26+; 15 still runs.
                          linkerSettings: [.unsafeFlags(["-Xlinker", "-platform_version", "-Xlinker", "macos",
                                                         "-Xlinker", "15.0", "-Xlinker", "26.0"])]),
        .testTarget(name: "CleanMyMicTests", dependencies: ["CleanMyMic"]),
    ]
)
