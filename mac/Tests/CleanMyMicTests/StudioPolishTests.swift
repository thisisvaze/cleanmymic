import Foundation
import Testing
@testable import CleanMyMic

@Test func studioPolishPreservesExactDigitalSilence() {
    var polish = StudioPolish(enabled: true)
    var silence = [Float](repeating: 0, count: 4800)
    polish.process(&silence)
    #expect(silence.allSatisfy { $0 == 0 })
}

@Test func peakLimiterPreventsCoreAudioClippingEvenWhenPolishIsOff() {
    for enabled in [false, true] {
        var polish = StudioPolish(enabled: enabled)
        var loud = (0..<4800).map { i in
            Float(1.4 * sin(2 * Double.pi * 440 * Double(i) / 48_000))
        }
        polish.process(&loud)
        let peak = loud.map(abs).max() ?? 0
        #expect(loud.allSatisfy { $0.isFinite })
        #expect(peak <= 0.985)
        #expect(peak > 0.5)
    }
}

@Test func highPassRemovesSubRumbleWhileKeepingSpeechMidband() {
    func steadyRMS(freq: Double) -> Double {
        var polish = StudioPolish(enabled: true)
        var signal = (0..<24_000).map { i in
            Float(0.05 * sin(2 * Double.pi * freq * Double(i) / 48_000))
        }
        polish.process(&signal)
        let tail = signal.dropFirst(12_000)
        return sqrt(tail.reduce(0.0) { $0 + Double($1 * $1) } / Double(tail.count))
    }
    let rumble = steadyRMS(freq: 30)
    let speech = steadyRMS(freq: 1000)
    let ratioDB = 20 * log10(rumble / max(speech, 1e-9))
    #expect(ratioDB < -12.0)  // 30 Hz attenuated well below 1 kHz speech
}

@Test func speechGateDoesNotBoostQuietNoiseFloor() {
    var polish = StudioPolish(enabled: true)
    // -55 dBFS quiet tone: below -45 dBFS speech gate, should not receive makeup boost
    let amp: Float = 0.0018
    var quiet = (0..<24_000).map { i in
        amp * Float(sin(2 * Double.pi * 1000 * Double(i) / 48_000))
    }
    let inRMS = sqrt(quiet.dropFirst(12_000).reduce(0.0) { $0 + Double($1 * $1) } / 12_000.0)
    polish.process(&quiet)
    let outRMS = sqrt(quiet.dropFirst(12_000).reduce(0.0) { $0 + Double($1 * $1) } / 12_000.0)
    #expect(outRMS <= inRMS * 1.1)
}

@Test func resetClearsFilterAndEnvelopeMemory() {
    var polish = StudioPolish(enabled: true)
    var burst = (0..<4800).map { i in Float(0.8 * sin(2 * Double.pi * 300 * Double(i) / 48_000)) }
    polish.process(&burst)
    polish.reset()
    var after = [Float](repeating: 0, count: 960)
    polish.process(&after)
    #expect(after.allSatisfy { $0 == 0 })
}
