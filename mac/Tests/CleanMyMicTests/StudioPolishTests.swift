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

@Test func inputPreampLiftsQuietMicsForModelAndRestoresTransparentOutput() {
    var preamp = InputPreamp(enabled: true, look: 2)
    // -60 dBFS quiet far-field mic speech (0.0014 peak / 0.001 RMS)
    var scaledPeak: Float = 0
    var restoredRMS = 0.0
    var inputRMS = 0.0
    var queue = [[Float]](repeating: [Float](repeating: 0, count: 480), count: 2)
    for hopIdx in 0..<800 {  // 8 s so the slow 2s/4s steady preamp settles without pumping
        let frame = (0..<480).map { j in
            let n = hopIdx * 480 + j
            return Float(0.0014 * sin(2 * Double.pi * 300 * Double(n) / 48_000))
        }
        let (scaled, outGain) = preamp.scaleInput(frame)
        queue.append(scaled)
        var delayed = queue.removeFirst()
        preamp.restoreOutput(&delayed, outGain: outGain)
        if hopIdx >= 700 {
            scaledPeak = max(scaledPeak, scaled.map(abs).max() ?? 0)
            restoredRMS += delayed.reduce(0.0) { $0 + Double($1 * $1) }
            inputRMS += frame.reduce(0.0) { $0 + Double($1 * $1) }
        }
    }
    // Model sees > 12 dB boosted signal (well inside DFN3's operating range), while output returns within 0.5 dB of original
    #expect(scaledPeak > 0.006)
    let ratioDB = abs(10 * log10(restoredRMS / max(inputRMS, 1e-12)))
    #expect(ratioDB < 0.5)
}

@Test func inputPreampLeavesNormalMicsAndDigitalSilenceUntouched() {
    var preamp = InputPreamp(enabled: true, look: 2)
    let silence = [Float](repeating: 0, count: 480)
    let (sOut, sGain) = preamp.scaleInput(silence)
    #expect(sGain == 1.0 && sOut.allSatisfy { $0 == 0 })

    // Normal -20 dBFS speech stays at 1.0x gain (no unnecessary pumping)
    for hopIdx in 0..<100 {
        let frame = (0..<480).map { j in
            let n = hopIdx * 480 + j
            return Float(0.14 * sin(2 * Double.pi * 400 * Double(n) / 48_000))
        }
        let (scaled, outGain) = preamp.scaleInput(frame)
        #expect(outGain == 1.0)
        #expect(scaled == frame)
    }
}
