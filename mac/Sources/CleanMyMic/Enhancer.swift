// One Core ML engine (from train/export.py) run the way model.stream() / dfn3.stream() do: 960-sample window,
// 480-sample hop, 48 kHz mono, 480 samples in and 480 out. Shared by the app and the mac/live.swift test tool.
// The window and spectrum compression come from the package's metadata: ours or DeepFilterNet3.
import Accelerate
import CoreML

nonisolated let win = 960, hop = 480, bins = 481

nonisolated final class Enhancer {
    let model: MLModel
    var state: [String: MLMultiArray] = [:]  // every input but x (GRU h, deep-filter buf), fed back from "<name>_out"
    let window: [Float], c: Float  // c: the model's spectrum exponent (1: plain spectrum)
    let look: Int  // frames the model looks ahead: its output lags by 1 + look hops
    var mix: Float = 1  // strength: 1 = the model's output, 0 = the raw input, time-aligned with it
    private var wasBypassed = false
    var buf = [Float](repeating: 0, count: win), tail = [Float](repeating: 0, count: hop), dry: [Float]
    let fwd = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(win), .FORWARD)!
    let inv = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(win), .INVERSE)!

    /// A compiled .mlmodelc (the app bundles these) or an .mlpackage (compiled here).
    init(_ url: URL) throws {
        model = try MLModel(contentsOf: url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url))
        for (name, desc) in model.modelDescription.inputDescriptionsByName where name != "x" {
            let a = try MLMultiArray(shape: desc.multiArrayConstraint!.shape, dataType: .float32)
            for i in 0..<a.count { a[i] = 0 }
            state[name] = a
        }
        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        c = Float(meta["compress"] ?? "0.3") ?? 0.3
        look = Int(meta["look"] ?? "0") ?? 0
        dry = [Float](repeating: 0, count: hop * look)
        window = meta["window"] == "vorbis"  // libDF's: sin(pi/2 sin^2(pi (n + 0.5) / N))
            ? (0..<win).map { Float(sin(0.5 * Double.pi * pow(sin(Double.pi * (Double($0) + 0.5) / Double(win)), 2))) }
            : (0..<win).map { sqrt(0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(win))) }
        // the first prediction prepares the model for the Neural Engine (seconds): pay that here, not on live audio
        let zero = try MLMultiArray(shape: [1, 1, NSNumber(value: 2 * bins)], dataType: .float32)
        for i in 0..<zero.count { zero[i] = 0 }
        _ = try autoreleasepool { try model.prediction(from: MLDictionaryFeatureProvider(dictionary: state.merging(["x": zero]) { $1 })) }
    }

    /// Standard (numpy-scaled) real DFT of 960 samples -> 481 bins. vDSP's zrop output is 2x and packs Nyquist in im[0].
    func rfft(_ x: [Float]) -> ([Float], [Float]) {
        let inR = (0..<hop).map { x[2 * $0] }, inI = (0..<hop).map { x[2 * $0 + 1] }
        var oR = [Float](repeating: 0, count: hop), oI = oR
        vDSP_DFT_Execute(fwd, inR, inI, &oR, &oI)
        var re = [Float](repeating: 0, count: bins), im = re
        re[0] = oR[0] / 2
        re[hop] = oI[0] / 2
        for k in 1..<hop { re[k] = oR[k] / 2; im[k] = oI[k] / 2 }
        return (re, im)
    }

    func irfft(_ re: [Float], _ im: [Float]) -> [Float] {
        let inR = Array(re.prefix(hop))
        var inI = Array(im.prefix(hop))
        inI[0] = re[hop]
        var oR = [Float](repeating: 0, count: hop), oI = oR
        vDSP_DFT_Execute(inv, inR, inI, &oR, &oI)
        var y = [Float](repeating: 0, count: win)
        for k in 0..<hop { y[2 * k] = oR[k] / Float(win); y[2 * k + 1] = oI[k] / Float(win) }
        return y
    }

    /// 480 samples in, 480 out.
    func process(_ input: [Float]) throws -> [Float] {
        buf.removeFirst(hop)
        buf.append(contentsOf: input)
        dry.append(contentsOf: buf[..<hop])  // the input that lines up with this output, `look` hops back
        let raw = Array(dry.prefix(hop))
        dry.removeFirst(hop)
        if mix == 0 { wasBypassed = true; return raw }
        if wasBypassed {  // don't replay stale overlap or recurrent state when cleanup resumes
            for a in state.values { for i in 0..<a.count { a[i] = 0 } }
            tail = [Float](repeating: 0, count: hop)
            wasBypassed = false
        }
        let (re, im) = rfft(zip(buf, window).map(*))
        let x = try MLMultiArray(shape: [1, 1, NSNumber(value: 2 * bins)], dataType: .float32)
        let xp = x.dataPointer.assumingMemoryBound(to: Float.self)
        for k in 0..<bins {
            let m = pow(hypot(re[k], im[k]) + 1e-8, c - 1)
            xp[k] = re[k] * m
            xp[bins + k] = im[k] * m
        }
        // Core ML hands its outputs (IOSurface-backed) out autoreleased: drained per frame, or a stream with no run
        // loop draining for it (the app's worker thread, file mode) runs out of IOSurfaces after a few minutes
        let y = try autoreleasepool {
            let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: state.merging(["x": x]) { $1 }))
            for name in state.keys { state[name] = out.featureValue(for: "\(name)_out")!.multiArrayValue! }
            return MLShapedArray<Float>(converting: out.featureValue(for: "y")!.multiArrayValue!).scalars
        }
        var yr = [Float](repeating: 0, count: bins), yi = yr
        for k in 0..<bins {
            let m = pow(hypot(y[k], y[bins + k]) + 1e-8, 1 / c - 1)
            yr[k] = y[k] * m
            yi[k] = y[bins + k] * m
        }
        let frame = zip(irfft(yr, yi), window).map(*)
        defer { tail = Array(frame[hop...]) }
        let wet = (0..<hop).map { tail[$0] + frame[$0] }
        return mix == 1 ? wet : (0..<hop).map { mix * wet[$0] + (1 - mix) * raw[$0] }
    }

    /// Checks the DFT scaling/sign against numpy conventions; everything downstream depends on it.
    func selfTest() {
        let x = (0..<win).map { sin(2 * Float.pi * 5 * Float($0) / Float(win)) + 0.3 }
        let (re, im) = rfft(x)
        assert(abs(im[5] + 480) < 0.01 && abs(re[0] - 288) < 0.01, "rfft scaling/sign wrong: \(re[0]) \(im[5])")
        let back = irfft(re, im)
        assert(zip(back, x).map { abs($0 - $1) }.max()! < 1e-4, "irfft(rfft(x)) != x")
    }
}
