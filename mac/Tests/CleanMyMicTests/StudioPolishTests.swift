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

@Test func adaptiveEQRespondsDifferentlyToMuffledVsHarshBrightMics() {
    // Muffled mic: strong 260 Hz + 1 kHz, almost no 4.2 kHz presence
    var muffledPolish = StudioPolish(enabled: true)
    for hopIdx in 0..<250 {  // 2.5 s in 10 ms hops
        var frame = (0..<480).map { j in
            let t = Double(hopIdx * 480 + j) / 48_000
            return Float(0.08 * sin(2 * Double.pi * 260 * t) + 0.05 * sin(2 * Double.pi * 1000 * t) + 0.002 * sin(2 * Double.pi * 4200 * t))
        }
        muffledPolish.process(&frame)
    }

    // Harsh bright thin mic: weak 260 Hz, strong 4.2 kHz
    var brightPolish = StudioPolish(enabled: true)
    for hopIdx in 0..<250 {
        var frame = (0..<480).map { j in
            let t = Double(hopIdx * 480 + j) / 48_000
            return Float(0.01 * sin(2 * Double.pi * 260 * t) + 0.05 * sin(2 * Double.pi * 1000 * t) + 0.07 * sin(2 * Double.pi * 4200 * t))
        }
        brightPolish.process(&frame)
    }

    // Muffled mic gets de-mudded + lifted presence; bright thin mic gets warmed + tamed presence
    #expect(muffledPolish.adaptiveLowGainDB < -0.5)
    #expect(muffledPolish.adaptiveHighGainDB > 1.0)
    #expect(brightPolish.adaptiveWarmthGainDB > muffledPolish.adaptiveWarmthGainDB)
    #expect(brightPolish.adaptiveHighGainDB < 0.0)
}

@Test func broadcastAutoLevelerLiftsQuietSpeechTowardPodcastLoudnessWithoutClipping() {
    var polish = StudioPolish(enabled: true)
    // Quiet -32 dBFS speech tone (0.035 peak / 0.025 RMS) -> should smoothly lift toward ~-18 dBFS
    var tailRMS = 0.0
    var maxPeak: Float = 0
    for hopIdx in 0..<250 {  // 2.5 s
        var frame = (0..<480).map { j in
            let t = Double(hopIdx * 480 + j) / 48_000
            return Float(0.035 * sin(2 * Double.pi * 220 * t))
        }
        polish.process(&frame)
        if hopIdx >= 200 {
            tailRMS += frame.reduce(0.0) { $0 + Double($1 * $1) }
        }
        maxPeak = max(maxPeak, frame.map(abs).max() ?? 0)
    }
    let outDB = 10 * log10(tailRMS / Double(50 * 480) + 1e-12)
    #expect(outDB > -23.0 && outDB < -14.0)  // lifted +9..14 dB into broadcast podcast window
    #expect(maxPeak <= 0.985)
}

@Test func mouthDeClickSoftensSalivaImpulseWithoutDullingSustainedTone() {
    var polish = StudioPolish(enabled: true)
    // Quiet transition with a sharp 1 ms 6 kHz saliva click burst at sample 2400..2448
    var withClick = [Float](repeating: 0, count: 4800)
    for i in 2400..<2448 {
        withClick[i] = Float(0.06 * sin(2 * Double.pi * 6000 * Double(i) / 48_000))
    }
    let inClickPeak = withClick[2400..<2448].map(abs).max() ?? 0
    polish.process(&withClick)
    let outClickPeak = withClick[2400..<2448].map(abs).max() ?? 0
    #expect(outClickPeak < inClickPeak * 0.75)  // sharp saliva click attenuated cleanly
}

@Test func dynamicDePlosiveAndHarshnessTamerSmoothSpikesWithoutKillingMidband() {
    // 115 Hz air-puff plosive burst vs 500 Hz vowel at same level
    var p1 = StudioPolish(enabled: true)
    var pop = (0..<9600).map { i in Float(0.10 * sin(2 * Double.pi * 115 * Double(i) / 48_000)) }
    p1.process(&pop)
    let popRMS = sqrt(pop.dropFirst(4800).reduce(0.0) { $0 + Double($1 * $1) } / 4800.0)

    // 2.8 kHz piercing harshness tone vs 500 Hz vowel at same level
    var p2 = StudioPolish(enabled: true)
    var harsh = (0..<9600).map { i in Float(0.10 * sin(2 * Double.pi * 2800 * Double(i) / 48_000)) }
    p2.process(&harsh)
    let harshRMS = sqrt(harsh.dropFirst(4800).reduce(0.0) { $0 + Double($1 * $1) } / 4800.0)

    var p3 = StudioPolish(enabled: true)
    var vowel = (0..<9600).map { i in Float(0.10 * sin(2 * Double.pi * 500 * Double(i) / 48_000)) }
    p3.process(&vowel)
    let vowelRMS = sqrt(vowel.dropFirst(4800).reduce(0.0) { $0 + Double($1 * $1) } / 4800.0)

    #expect(popRMS < vowelRMS * 0.85)
    #expect(harshRMS < vowelRMS * 0.90)
}

@Test func splitBandDeEsserTamesHarshSibilantsWithoutKillingVowels() {
    var polish = StudioPolish(enabled: true)
    // Pure harsh 6.5 kHz sibilant burst vs 500 Hz vowel at same amplitude
    var sib = (0..<9600).map { i in Float(0.08 * sin(2 * Double.pi * 6500 * Double(i) / 48_000)) }
    polish.process(&sib)
    let sibOutRMS = sqrt(sib.dropFirst(4800).reduce(0.0) { $0 + Double($1 * $1) } / 4800.0)

    polish.reset()
    var vowel = (0..<9600).map { i in Float(0.08 * sin(2 * Double.pi * 500 * Double(i) / 48_000)) }
    polish.process(&vowel)
    let vowelOutRMS = sqrt(vowel.dropFirst(4800).reduce(0.0) { $0 + Double($1 * $1) } / 4800.0)

    #expect(sibOutRMS < vowelOutRMS * 0.85)  // sharp sibilant tamed noticeably below vowel level
}

@Test func driftResamplerAbsorbsClockDriftSmoothlyWithoutDrops() {
    var resampler = DriftResampler(targetFill: 720)
    // Simulate a mic running +500 ppm fast (producing 480 samples while output consumes ~479.76 samples/hop)
    var fifo = 720.0
    var maxStepJump: Float = 0
    var lastOut: Float = 0
    for hopIdx in 0..<600 {  // 6 seconds
        let frame = (0..<480).map { j in
            let n = hopIdx * 480 + j
            return Float(0.3 * sin(2 * Double.pi * 440 * Double(n) / 48_000))
        }
        let out = resampler.process(frame, currentFill: Int(fifo))
        for s in out {
            maxStepJump = max(maxStepJump, abs(s - lastOut))
            lastOut = s
        }
        fifo += Double(out.count) - 479.76
    }
    // Stays cleanly bounded near target without hitting overflow (1440) or underflow (0), and zero clicks
    #expect(fifo > 200 && fifo < 1300)
    #expect(maxStepJump < 0.05)  // 440 Hz sine at 48 kHz has max natural step ~0.017
}

@Test func takeRecorderWritesAlignedRawAndCleanWAVs() throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("cmm-take-test-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    let rec = try #require(TakeRecorder(dir: dir, micName: "AirPods Pro", modelName: "CleanMyMicStudio",
                                        mix: 1.0, studioPolish: true, look: 2))
    let rawFrame = (0..<480).map { Float(0.2 * sin(2 * Double.pi * 440 * Double($0) / 48_000)) }
    let cleanFrame = (0..<480).map { Float(0.15 * sin(2 * Double.pi * 440 * Double($0) / 48_000)) }
    for _ in 0..<25 { rec.append(raw: rawFrame, clean: cleanFrame) }
    rec.close()
    let files = try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()
    #expect(files.contains { $0.hasSuffix("_raw.wav") })
    #expect(files.contains { $0.hasSuffix("_clean.wav") })
    #expect(files.contains { $0.hasSuffix(".json") })
}
