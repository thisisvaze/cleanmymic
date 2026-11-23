// Pro podcast/broadcast DSP around the neural model (48 kHz mono, zero extra latency):
// - InputPreamp (pre-model, transparently undone post-model): smoothly lifts very quiet mics (-65..-35 dBFS)
//   into the neural model's -24 dBFS sweet spot so DFN3's running norms never starve and duck quiet speech.
// - StudioPolish (post-model, when `enabled` is on):
//   1. 80 Hz Butterworth high-pass (removes desk thumps, AC rumble, DC)
//   2. Real-time Mouth De-click (softens sub-2 ms 3.2+ kHz wet saliva/lip-smack impulses) + Dynamic 115 Hz De-plosive ("p"/"b" pops)
//   3. SM7B-style chest-warmth low shelf (125 Hz) + speech-gated mic-adaptive EQ (300 Hz de-mud, 3.4 kHz presence, 10.5 kHz air)
//   4. 2.8 kHz Dynamic Harshness Tamer (smooths piercing "r"/"ee"/"ch" resonances) + 6.5 kHz Split-Band De-esser ("s"/"sh")
//   5. Two-stage serial broadcast dynamics: speech-gated -18 dBFS auto-leveler (never lifts pauses) + fast 3.2:1 vocal compressor
//   6. Subtle analog/tube harmonic warmth exciter (DC-blocked 2nd/3rd harmonics to thicken thin AirPods/laptop mics)
//   7. Peak safety limiter (-1 dBTP) when `limitPeaks` is on (prevents Core ML overshoots clipping Core Audio)
import Foundation

/// Transparent pre-model mic preamp that prevents quiet-mic voice dropouts (e.g. far-field rooms or leaning back).
/// Scales input frames up toward -24 dBFS for the neural model, then divides output frames by the time-aligned
/// ramped gain (`look` hops delayed) so perceived loudness stays unchanged without pumping.
nonisolated struct InputPreamp {
    var enabled: Bool
    private let look: Int
    private var env = 0.05  // start neutral (-26 dBFS): normal mics stay at 1.0x immediately
    private var gain = 1.0
    private var prevOutGain = 1.0
    private var gainHistory: [Double]

    // Per-10ms-hop constants at 48 kHz (hop = 480):
    // Very slow release (4 s) & rise (2 s) keep gain rock-steady mid-sentence with < 0.02 dB/hop movement,
    // eliminating any audible fluctuation while lifting quiet far-field mics into the neural model's sweet spot.
    private let envAttack = 1.0 - exp(-480.0 / (0.04 * 48_000))   // 40 ms attack on louder speech
    private let envRelease = 1.0 - exp(-480.0 / (4.0 * 48_000))   // 4 s slow release -> zero mid-sentence pumping
    private let gainDown = 1.0 - exp(-480.0 / (0.15 * 48_000))    // 150 ms smooth back-off (hard peak cap still guards spikes)
    private let gainUp = 1.0 - exp(-480.0 / (2.0 * 48_000))       // 2 s smooth rise -> stays slower than DFN3 ERB norm
    private let maxGain = 40.0                                     // up to +32 dB for -60 dBFS far-field mics

    init(enabled: Bool = false, look: Int = 2) {
        self.enabled = enabled
        self.look = max(0, look)
        self.gainHistory = [Double](repeating: 1.0, count: self.look + 1)
    }

    mutating func reset() {
        env = 0.05
        gain = 1.0
        prevOutGain = 1.0
        gainHistory = [Double](repeating: 1.0, count: look + 1)
    }

    /// Scales a 480-sample input frame for the neural model and returns the time-aligned gain to undo on output.
    mutating func scaleInput(_ input: [Float]) -> (scaled: [Float], outGain: Double) {
        guard enabled else { return (input, 1.0) }
        var sumSq = 0.0
        var peak = 0.0
        for s in input {
            let v = Double(s)
            sumSq += v * v
            let av = abs(v)
            if av > peak { peak = av }
        }
        let rms = sqrt(sumSq / Double(max(1, input.count)))
        if rms > 1e-5 {  // ignore digital silence
            let alpha = rms > env ? envAttack : envRelease
            env += alpha * (rms - env)
            let maxSafe = 0.85 / max(peak, 1e-6)
            // Deadzone above -27 dBFS (0.045): normal/loud mics stay untouched at 1.0x
            let desired = env < 0.045 ? (0.063 / max(env, 1e-4)) : 1.0
            let target = min(max(desired, 1.0), min(maxGain, maxSafe))
            let gAlpha = target < gain ? gainDown : gainUp
            gain = min(gain + gAlpha * (target - gain), maxSafe)
        }
        gainHistory.append(gain)
        let outGain = gainHistory.removeFirst()
        if gain == 1.0 { return (input, outGain) }
        let g = Float(gain)
        return (input.map { $0 * g }, outGain)
    }

    /// Transparently undoes the time-aligned pre-gain on the 480-sample model output with click-free linear ramp.
    mutating func restoreOutput(_ output: inout [Float], outGain: Double) {
        guard enabled else { return }
        if prevOutGain == 1.0 && outGain == 1.0 { return }
        let n = output.count
        let denom = Double(max(1, n - 1))
        for i in 0..<n {
            let g = prevOutGain + (outGain - prevOutGain) * (Double(i) / denom)
            output[i] = Float(Double(output[i]) / max(g, 1e-6))
        }
        prevOutGain = outGain
    }
}

/// Transparent micro-rate adaptive resampler for cross-device clock drift (e.g. AirPods / USB mic vs virtual cable).
/// Nudges output playback rate by at most ±0.2% (±3.5 cents, completely inaudible) to keep the output FIFO centered
/// around its target fill without ever dropping or zero-padding samples during normal clock drift.
nonisolated struct DriftResampler {
    private var prevSample: Float = 0
    private var phase: Double = 0
    private var fillEMA: Double
    private let targetFill: Double
    private let maxSkew = 0.002  // ±2000 ppm max rate nudge (inaudible, ~20x typical crystal drift)

    init(targetFill: Int = 720) {
        self.targetFill = Double(targetFill)
        self.fillEMA = Double(targetFill)
    }

    mutating func reset() {
        prevSample = 0
        phase = 0
        fillEMA = targetFill
    }

    /// Resamples one output frame based on current FIFO fill level so the FIFO gently converges to `targetFill`.
    mutating func process(_ frame: [Float], currentFill: Int) -> [Float] {
        guard !frame.isEmpty else { return frame }
        fillEMA += 0.02 * (Double(currentFill) - fillEMA)  // ~0.5 s smoothing at 100 Hz
        let error = fillEMA - targetFill
        // Deadzone within ±120 samples (±2.5 ms): exact 1.0x bit-identical passthrough when clocks agree
        if abs(error) < 120 && phase == 0 {
            prevSample = frame[frame.count - 1]
            return frame
        }
        // Positive error (FIFO growing) -> step faster (>1.0) to produce slightly fewer samples; negative -> slightly more
        let step = 1.0 + min(maxSkew, max(-maxSkew, error * 2.5e-6))
        let n = frame.count
        var out: [Float] = []
        out.reserveCapacity(n + 2)
        var p = phase
        while p < Double(n) {
            let idx = Int(floor(p))
            let frac = Float(p - Double(idx))
            let s0 = idx <= 0 ? prevSample : frame[idx - 1]
            let s1 = idx < n ? frame[idx] : frame[n - 1]
            out.append(s0 + frac * (s1 - s0))
            p += step
        }
        phase = p - Double(n)
        prevSample = frame[n - 1]
        return out
    }
}

nonisolated struct Biquad {
    private var b0: Double, b1: Double, b2: Double, a1: Double, a2: Double
    private var z1 = 0.0, z2 = 0.0

    init(b0: Double, b1: Double, b2: Double, a0: Double, a1: Double, a2: Double) {
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
    }

    mutating func updateCoefficients(from other: Biquad) {
        b0 = other.b0
        b1 = other.b1
        b2 = other.b2
        a1 = other.a1
        a2 = other.a2
    }

    static func highPass(frequency: Double, sampleRate: Double = 48_000, q: Double = 0.7071067811865476) -> Biquad {
        let w0 = 2 * Double.pi * frequency / sampleRate
        let c = cos(w0), alpha = sin(w0) / (2 * q)
        return Biquad(b0: (1 + c) / 2, b1: -(1 + c), b2: (1 + c) / 2,
                      a0: 1 + alpha, a1: -2 * c, a2: 1 - alpha)
    }

    static func bandPass(frequency: Double, q: Double, sampleRate: Double = 48_000) -> Biquad {
        let w0 = 2 * Double.pi * frequency / sampleRate
        let c = cos(w0), alpha = sin(w0) / (2 * q)
        return Biquad(b0: alpha, b1: 0, b2: -alpha,
                      a0: 1 + alpha, a1: -2 * c, a2: 1 - alpha)
    }

    static func peaking(frequency: Double, gainDB: Double, q: Double, sampleRate: Double = 48_000) -> Biquad {
        let a = pow(10, gainDB / 40)
        let w0 = 2 * Double.pi * frequency / sampleRate
        let c = cos(w0), alpha = sin(w0) / (2 * q)
        return Biquad(b0: 1 + alpha * a, b1: -2 * c, b2: 1 - alpha * a,
                      a0: 1 + alpha / a, a1: -2 * c, a2: 1 - alpha / a)
    }

    static func lowShelf(frequency: Double, gainDB: Double, slope: Double = 0.7, sampleRate: Double = 48_000) -> Biquad {
        let a = pow(10, gainDB / 40)
        let w0 = 2 * Double.pi * frequency / sampleRate
        let c = cos(w0), s = sin(w0)
        let alpha = (s / 2) * sqrt((a + 1 / a) * (1 / slope - 1) + 2)
        let twoSqrtAAlpha = 2 * sqrt(a) * alpha
        return Biquad(
            b0: a * ((a + 1) - (a - 1) * c + twoSqrtAAlpha),
            b1: 2 * a * ((a - 1) - (a + 1) * c),
            b2: a * ((a + 1) - (a - 1) * c - twoSqrtAAlpha),
            a0: (a + 1) + (a - 1) * c + twoSqrtAAlpha,
            a1: -2 * ((a - 1) + (a + 1) * c),
            a2: (a + 1) + (a - 1) * c - twoSqrtAAlpha
        )
    }

    static func highShelf(frequency: Double, gainDB: Double, slope: Double = 0.7, sampleRate: Double = 48_000) -> Biquad {
        let a = pow(10, gainDB / 40)
        let w0 = 2 * Double.pi * frequency / sampleRate
        let c = cos(w0), s = sin(w0)
        let alpha = (s / 2) * sqrt((a + 1 / a) * (1 / slope - 1) + 2)
        let twoSqrtAAlpha = 2 * sqrt(a) * alpha
        return Biquad(
            b0: a * ((a + 1) + (a - 1) * c + twoSqrtAAlpha),
            b1: -2 * a * ((a - 1) + (a + 1) * c),
            b2: a * ((a + 1) + (a - 1) * c - twoSqrtAAlpha),
            a0: (a + 1) - (a - 1) * c + twoSqrtAAlpha,
            a1: 2 * ((a - 1) - (a + 1) * c),
            a2: (a + 1) - (a - 1) * c - twoSqrtAAlpha
        )
    }

    mutating func process(_ x: Double) -> Double {
        let y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        return y
    }

    mutating func reset() {
        z1 = 0
        z2 = 0
    }
}

nonisolated struct StudioPolish {
    var enabled: Bool
    var limitPeaks: Bool
    private let sampleRate: Double
    private var hpf: Biquad
    private var warmthShelf: Biquad   // 125 Hz chest warmth ("SM7B" proximity weight above 80 Hz HPF)
    private var lowBandEQ: Biquad     // 300 Hz boxy-mud control
    private var highBandEQ: Biquad    // 3.4 kHz vocal clarity / presence
    private var air: Biquad           // 10.5 kHz studio air shelf
    private var dcBlocker: Biquad     // removes any tiny DC offset from asymmetric tube warmth
    // Sidechain 3-band speech-balance analyzers + vocal cleanup detectors (mouth de-click, de-plosive, harshness, de-esser)
    private var anaLow: Biquad
    private var anaMid: Biquad
    private var anaHigh: Biquad
    private var clickLP = 0.0         // complementary 1-pole crossover (~2.5 kHz) so low + hi == x with zero phase error
    private var plosiveFilter: Biquad
    private var harshFilter: Biquad
    private var sibFilter: Biquad
    // Slow speech-gated spectral tilt trackers (frozen during pauses/silence)
    private var lowPow = 0.0
    private var midPow = 0.0
    private var highPow = 0.0
    private var speechSamples = 0
    private var currentWarmthDB = 1.8
    private var currentLowDB = -1.6
    private var currentHighDB = 1.4
    private var currentAirDB = 1.0
    // Mouth de-click, dynamic de-plosive, 2.8 kHz harshness tamer, and 6.5 kHz de-esser envelopes
    private var clickFast = 0.0
    private var clickSlow = 0.0
    private var clickAtten = 0.0
    private var plosiveEnv = 0.0
    private var midEnv = 0.0
    private var plosiveAtten = 0.0
    private var harshEnv = 0.0
    private var harshAtten = 0.0
    private var sibEnv = 0.0
    private var deEssAtten = 0.0
    // Stage 1: Speech-gated broadcast auto-leveler (targets -16 dBFS active speech RMS ≈ -16 LUFS broadcast standard)
    private var rmsEnv = 0.0
    private var speechLevelEnv = 0.06   // starts near -24 dBFS; frozen on pauses so gain stays rock-steady word-to-word
    private var levelGain = 1.0
    private var seenSpeech = false
    // Stage 2: Fast serial vocal compressor (tames sharp transients & laughs right after leveler)
    private var fastEnv = 0.0
    private var compGain = 1.0
    private var limEnv = 0.0

    // Speech-gated dynamics & vocal-cleanup constants (48 kHz)
    private let rmsAlpha = exp(-1.0 / (0.025 * 48_000))        // 25 ms RMS detector
    private let clickLPAlpha = 1.0 - exp(-2.0 * Double.pi * 2500.0 / 48_000)  // 2.5 kHz complementary split
    private let clickFastAttack = exp(-1.0 / (0.00004 * 48_000)) // 0.04 ms (~2-sample) instant catch for saliva/lip clicks
    private let clickFastRelease = exp(-1.0 / (0.0025 * 48_000)) // 2.5 ms smooth recovery
    private let clickSlowAlpha = exp(-1.0 / (0.018 * 48_000))    // 18 ms local high-band context
    private let plosiveAttack = exp(-1.0 / (0.003 * 48_000))     // 3 ms "p"/"b" pop catch
    private let plosiveRelease = exp(-1.0 / (0.035 * 48_000))    // 35 ms recovery
    private let midEnvAlpha = exp(-1.0 / (0.015 * 48_000))       // 15 ms midband voice reference
    private let harshAttack = exp(-1.0 / (0.004 * 48_000))       // 4 ms harsh resonance catch
    private let harshRelease = exp(-1.0 / (0.045 * 48_000))      // 45 ms smooth release
    private let fastAttack = exp(-1.0 / (0.005 * 48_000))      // 5 ms FET-style transient catch
    private let fastRelease = exp(-1.0 / (0.100 * 48_000))     // 100 ms musical release
    private let lvlEnvAttack = exp(-1.0 / (0.15 * 48_000))     // 150 ms level tracker attack
    private let lvlEnvRelease = exp(-1.0 / (0.70 * 48_000))    // 700 ms smooth level tracker release
    private let lvlGainDown = exp(-1.0 / (0.08 * 48_000))      // 80 ms gain pull-back when speaking louder
    private let lvlGainUp = exp(-1.0 / (0.20 * 48_000))        // 200 ms crisp broadcast level ride up
    private let compAttack = exp(-1.0 / (0.0025 * 48_000))     // 2.5 ms fast vocal compressor attack (catches consonant snaps cleanly)
    private let compRelease = exp(-1.0 / (0.110 * 48_000))     // 110 ms smooth release
    private let sibAttack = exp(-1.0 / (0.003 * 48_000))       // 3 ms sibilant catch
    private let sibRelease = exp(-1.0 / (0.035 * 48_000))      // 35 ms transparent release
    private let tiltAlpha = 1.0 - exp(-1.0 / (3.0 * 48_000))   // 3 s speech-gated mic balance EMA
    private let eqSmooth = 0.06                                 // gentle per-frame (~10 ms) EQ coeff glide
    // -1 dBTP peak safety limiter constants (48 kHz)
    private let ceiling = 0.8912509381337456                   // 10^(-1 / 20)
    private let limAttack = exp(-1.0 / (0.0004 * 48_000))      // 0.4 ms fast peak catch
    private let limRelease = exp(-1.0 / (0.045 * 48_000))      // 45 ms transparent release

    /// Exposed for diagnostics/tests: current adaptive EQ gains in dB.
    var adaptiveLowGainDB: Double { currentLowDB }
    var adaptiveHighGainDB: Double { currentHighDB }
    var adaptiveWarmthGainDB: Double { currentWarmthDB }

    init(enabled: Bool = true, limitPeaks: Bool = true, sampleRate: Double = 48_000) {
        self.enabled = enabled
        self.limitPeaks = limitPeaks
        self.sampleRate = sampleRate
        self.hpf = .highPass(frequency: 80, sampleRate: sampleRate)
        self.warmthShelf = .lowShelf(frequency: 125, gainDB: 1.8, slope: 0.75, sampleRate: sampleRate)
        self.lowBandEQ = .peaking(frequency: 300, gainDB: -1.6, q: 1.0, sampleRate: sampleRate)
        self.highBandEQ = .peaking(frequency: 3400, gainDB: 1.4, q: 0.85, sampleRate: sampleRate)
        self.air = .highShelf(frequency: 10_500, gainDB: 1.0, slope: 0.7, sampleRate: sampleRate)
        self.dcBlocker = .highPass(frequency: 25, sampleRate: sampleRate)
        self.anaLow = .bandPass(frequency: 280, q: 0.8, sampleRate: sampleRate)
        self.anaMid = .bandPass(frequency: 1000, q: 0.8, sampleRate: sampleRate)
        self.anaHigh = .bandPass(frequency: 4200, q: 0.8, sampleRate: sampleRate)
        self.plosiveFilter = .bandPass(frequency: 115, q: 1.2, sampleRate: sampleRate)
        self.harshFilter = .bandPass(frequency: 2800, q: 1.6, sampleRate: sampleRate)
        self.sibFilter = .bandPass(frequency: 6500, q: 1.4, sampleRate: sampleRate)
    }

    mutating func reset() {
        hpf.reset()
        warmthShelf.reset()
        lowBandEQ.reset()
        highBandEQ.reset()
        air.reset()
        dcBlocker.reset()
        anaLow.reset()
        anaMid.reset()
        anaHigh.reset()
        clickLP = 0
        plosiveFilter.reset()
        harshFilter.reset()
        sibFilter.reset()
        lowPow = 0
        midPow = 0
        highPow = 0
        speechSamples = 0
        currentWarmthDB = 1.8
        currentLowDB = -1.6
        currentHighDB = 1.4
        currentAirDB = 1.0
        warmthShelf.updateCoefficients(from: .lowShelf(frequency: 125, gainDB: currentWarmthDB, slope: 0.75, sampleRate: sampleRate))
        lowBandEQ.updateCoefficients(from: .peaking(frequency: 300, gainDB: currentLowDB, q: 1.0, sampleRate: sampleRate))
        highBandEQ.updateCoefficients(from: .peaking(frequency: 3400, gainDB: currentHighDB, q: 0.85, sampleRate: sampleRate))
        air.updateCoefficients(from: .highShelf(frequency: 10_500, gainDB: currentAirDB, slope: 0.7, sampleRate: sampleRate))
        clickFast = 0
        clickSlow = 0
        clickAtten = 0
        plosiveEnv = 0
        midEnv = 0
        plosiveAtten = 0
        harshEnv = 0
        harshAtten = 0
        sibEnv = 0
        deEssAtten = 0
        rmsEnv = 0
        speechLevelEnv = 0.06
        levelGain = 1
        seenSpeech = false
        fastEnv = 0
        compGain = 1
        limEnv = 0
    }

    private mutating func updateAdaptiveEQ() {
        // Wait for ~0.25 s of active speech before nudging away from neutral podcast startup curve
        guard speechSamples >= 12_000, midPow > 1e-9 else { return }
        let lowVsMidDB = 10 * log10((lowPow + 1e-12) / (midPow + 1e-12))
        let highVsMidDB = 10 * log10((highPow + 1e-12) / (midPow + 1e-12))
        // SM7B-style broadcast curve: rich 125 Hz chest warmth + clean 300 Hz de-mud + balanced 3.4 kHz presence
        let targetLowDB = min(1.0, max(-3.0, (-1.0 - lowVsMidDB) * 0.35))
        let targetWarmthDB = min(2.8, max(0.5, 1.8 + (-1.0 - lowVsMidDB) * 0.15))
        let targetHighDB = min(2.8, max(-2.0, (-7.5 - highVsMidDB) * 0.35))
        let targetAirDB = targetHighDB > 0 ? min(1.8, 0.6 + 0.4 * targetHighDB) : max(-1.0, 0.4 * targetHighDB)

        currentWarmthDB += eqSmooth * (targetWarmthDB - currentWarmthDB)
        currentLowDB += eqSmooth * (targetLowDB - currentLowDB)
        currentHighDB += eqSmooth * (targetHighDB - currentHighDB)
        currentAirDB += eqSmooth * (targetAirDB - currentAirDB)

        warmthShelf.updateCoefficients(from: .lowShelf(frequency: 125, gainDB: currentWarmthDB, slope: 0.75, sampleRate: sampleRate))
        lowBandEQ.updateCoefficients(from: .peaking(frequency: 300, gainDB: currentLowDB, q: 1.0, sampleRate: sampleRate))
        highBandEQ.updateCoefficients(from: .peaking(frequency: 3400, gainDB: currentHighDB, q: 0.85, sampleRate: sampleRate))
        air.updateCoefficients(from: .highShelf(frequency: 10_500, gainDB: currentAirDB, slope: 0.7, sampleRate: sampleRate))
    }

    /// Processes a 48 kHz frame in place: peak safety always runs when `limitPeaks` is on; full podcast mastering runs when `enabled`.
    mutating func process(_ samples: inout [Float]) {
        guard enabled || limitPeaks else { return }
        if enabled {
            updateAdaptiveEQ()
        }
        for i in samples.indices {
            var x = Double(samples[i])
            if !x.isFinite { x = 0 }

            if enabled {
                x = hpf.process(x)

                // Track speech RMS + sidechain mic balance strictly on active speech (> -40 dBFS)
                rmsEnv = rmsAlpha * rmsEnv + (1 - rmsAlpha) * (x * x)
                let rms = sqrt(rmsEnv)
                let aLow = anaLow.process(x)
                let aMid = anaMid.process(x)
                let aHigh = anaHigh.process(x)
                midEnv = midEnvAlpha * midEnv + (1 - midEnvAlpha) * abs(aMid)
                if rms > 0.01 {  // > -40 dBFS: active speech only (pauses never drift EQ)
                    lowPow += tiltAlpha * (aLow * aLow - lowPow)
                    midPow += tiltAlpha * (aMid * aMid - midPow)
                    highPow += tiltAlpha * (aHigh * aHigh - highPow)
                    if speechSamples < 480_000 { speechSamples += 1 }
                }

                // 0a) Real-time Mouth De-click: complementary 2.5 kHz split softens sub-2 ms saliva/lip clicks cleanly
                clickLP += clickLPAlpha * (x - clickLP)
                let hiClick = x - clickLP
                let aHi = abs(hiClick)
                let cfCoeff = aHi > clickFast ? clickFastAttack : clickFastRelease
                clickFast = cfCoeff * clickFast + (1 - cfCoeff) * aHi
                clickSlow = clickSlowAlpha * clickSlow + (1 - clickSlowAlpha) * aHi
                let targetClick: Double
                if clickFast > 0.004 && clickFast < 0.14 && rms < 0.09 && clickFast > 3.2 * max(clickSlow, 0.0005) {
                    let ratio = (clickFast / max(clickSlow, 0.0005) - 3.2) / 3.0
                    targetClick = min(0.78, max(0.0, ratio * 0.78))
                } else {
                    targetClick = 0
                }
                let caCoeff = targetClick > clickAtten ? 0.25 : clickFastRelease
                clickAtten = caCoeff * clickAtten + (1 - caCoeff) * targetClick
                x = clickLP + (1.0 - clickAtten) * hiClick

                // 0b) Dynamic 115 Hz De-plosive: catches sudden "p"/"b" air-puff pops while preserving chest warmth on vowels
                let plosive = plosiveFilter.process(x)
                let aPlo = abs(plosive)
                let pCoeff = aPlo > plosiveEnv ? plosiveAttack : plosiveRelease
                plosiveEnv = pCoeff * plosiveEnv + (1 - pCoeff) * aPlo
                let targetPlo: Double
                if plosiveEnv > 0.025 && plosiveEnv > 1.35 * max(midEnv, 0.008) {
                    let excess = (plosiveEnv - 1.35 * max(midEnv, 0.008)) / max(plosiveEnv, 1e-6)
                    targetPlo = min(0.65, excess * 0.85)
                } else {
                    targetPlo = 0
                }
                let paCoeff = targetPlo > plosiveAtten ? plosiveAttack : plosiveRelease
                plosiveAtten = paCoeff * plosiveAtten + (1 - paCoeff) * targetPlo
                x -= plosiveAtten * plosive

                // 1) SM7B warmth low shelf (125 Hz) + mic-adaptive de-mud (300 Hz), presence (3.4 kHz), air (10.5 kHz)
                x = warmthShelf.process(x)
                x = lowBandEQ.process(x)
                x = highBandEQ.process(x)
                x = air.process(x)

                // 2a) 2.8 kHz Dynamic Harshness Tamer ("mini-Soothe2"): smooths piercing "r"/"ee"/"ch" & nasal spikes
                let harsh = harshFilter.process(x)
                let aHarsh = abs(harsh)
                let hCoeff = aHarsh > harshEnv ? harshAttack : harshRelease
                harshEnv = hCoeff * harshEnv + (1 - hCoeff) * aHarsh
                let targetHarsh: Double
                if harshEnv > 0.015 && harshEnv > rms * 0.42 {
                    let excess = (harshEnv - rms * 0.42) / max(harshEnv, 1e-6)
                    targetHarsh = min(0.42, excess * 0.65)  // up to ~4.7 dB dynamic harshness smoothing
                } else {
                    targetHarsh = 0
                }
                let haCoeff = targetHarsh > harshAtten ? harshAttack : harshRelease
                harshAtten = haCoeff * harshAtten + (1 - haCoeff) * targetHarsh
                x -= harshAtten * harsh

                // 2b) Transparent split-band vocal de-esser (6.5 kHz): trims sharp sibilants exceeding speech body
                let sib = sibFilter.process(x)
                let asib = abs(sib)
                let sCoeff = asib > sibEnv ? sibAttack : sibRelease
                sibEnv = sCoeff * sibEnv + (1 - sCoeff) * asib
                let targetDeEss: Double
                if sibEnv > 0.012 && sibEnv > rms * 0.35 {
                    let excess = (sibEnv - rms * 0.35) / max(sibEnv, 1e-6)
                    targetDeEss = min(0.48, excess * 0.68)  // up to ~5.7 dB split-band sibilant trim
                } else {
                    targetDeEss = 0
                }
                let dCoeff = targetDeEss > deEssAtten ? sibAttack : sibRelease
                deEssAtten = dCoeff * deEssAtten + (1 - dCoeff) * targetDeEss
                x -= deEssAtten * sib

                // 3) Stage A — Speech-gated broadcast auto-leveler (targets -17.5 dBFS active speech ≈ -16..-18 LUFS podcast)
                // Updates ONLY on clear speech (> -36 dBFS) so soft breaths/onsets never over-boost, and HOLDS rock-steady
                // across brief word pauses (zero onset pumping), staying at 1.0x until real speech is heard.
                if rms > 0.0158 {  // > -36 dBFS: clear vocal body (not quiet breath/room floor)
                    if !seenSpeech {
                        speechLevelEnv = rms
                        seenSpeech = true
                    } else {
                        let lCoeff = rms > speechLevelEnv ? lvlEnvAttack : lvlEnvRelease
                        speechLevelEnv = lCoeff * speechLevelEnv + (1 - lCoeff) * rms
                    }
                }
                let targetLevelGain: Double
                if seenSpeech {
                    // Hold steady mic gain across speech & short pauses (up to +12 dB / -6 dB, zero limiter slam)
                    targetLevelGain = min(4.0, max(0.5, 0.135 / max(speechLevelEnv, 0.018)))
                } else {
                    targetLevelGain = 1.0
                }
                let gCoeff = targetLevelGain < levelGain ? lvlGainDown : lvlGainUp
                levelGain = gCoeff * levelGain + (1 - gCoeff) * targetLevelGain
                x *= levelGain

                // 4) Stage B — Fast FET/opto broadcast vocal compressor (tames sharp syllables cleanly before limiter)
                let axFast = abs(x)
                let fCoeff = axFast > fastEnv ? fastAttack : fastRelease
                fastEnv = fCoeff * fastEnv + (1 - fCoeff) * axFast
                let compTarget: Double
                if fastEnv < 0.01 || !seenSpeech {
                    compTarget = 1.0
                } else {
                    let levelDB = 20 * log10(fastEnv)
                    let thresholdDB = -14.0
                    let kneeDB = 6.0
                    let ratio = 3.2
                    let over = levelDB - thresholdDB
                    let reductionDB: Double
                    if over <= -kneeDB / 2 {
                        reductionDB = 0
                    } else if over < kneeDB / 2 {
                        let t = over + kneeDB / 2
                        reductionDB = (1 / ratio - 1) * (t * t) / (2 * kneeDB)
                    } else {
                        reductionDB = (1 / ratio - 1) * over
                    }
                    compTarget = pow(10, (reductionDB + 1.0) / 20)
                }
                let cCoeff = compTarget < compGain ? compAttack : compRelease
                compGain = cCoeff * compGain + (1 - cCoeff) * compTarget
                x *= compGain

                // 5) Subtle analog/tube harmonic warmth (RødeCaster/Neve-style: gentle 2nd+3rd harmonics, 18% wet, DC-blocked)
                if x != 0 {
                    let drive = 1.35
                    let sat = (tanh(drive * x) / drive) + 0.025 * x * x
                    x = dcBlocker.process(0.82 * x + 0.18 * sat)
                }
            }

            if limitPeaks {
                // -1 dBTP peak safety limiter + soft-clip knee so output never clips Core Audio
                let ax = abs(x)
                let lCoeff = ax > limEnv ? limAttack : limRelease
                limEnv = lCoeff * limEnv + (1 - lCoeff) * ax
                if limEnv > ceiling {
                    x *= ceiling / limEnv
                }
                if abs(x) > ceiling {
                    let sign = x >= 0 ? 1.0 : -1.0
                    let excess = abs(x) - ceiling
                    let headroom = 0.98 - ceiling
                    x = sign * (ceiling + headroom * tanh(excess / headroom))
                }
            }
            samples[i] = Float(x)
        }
    }
}
