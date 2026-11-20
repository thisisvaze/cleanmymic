// Lightweight broadcast DSP around the neural model (48 kHz mono, zero extra latency):
// - InputPreamp (pre-model, transparently undone post-model): smoothly lifts very quiet mics (-65..-35 dBFS)
//   into the neural model's -24 dBFS sweet spot so DFN3's running norms never starve and duck quiet speech.
// - StudioPolish (post-model):
//   1. 75 Hz Butterworth high-pass (removes desk thumps, plosives, DC)
//   2. Subtle studio vocal EQ (gentle 280 Hz de-mud, 3.2 kHz presence, 10 kHz air) when `enabled` is on
//   3. Speech-gated soft-knee leveler/compressor (never boosts silence/pauses) when `enabled` is on
//   4. Peak safety limiter (-1 dBTP) when `limitPeaks` is on (prevents Core ML overshoots clipping Core Audio)
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

nonisolated struct Biquad {
    private let b0: Double, b1: Double, b2: Double, a1: Double, a2: Double
    private var z1 = 0.0, z2 = 0.0

    init(b0: Double, b1: Double, b2: Double, a0: Double, a1: Double, a2: Double) {
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
    }

    static func highPass(frequency: Double, sampleRate: Double = 48_000, q: Double = 0.7071067811865476) -> Biquad {
        let w0 = 2 * Double.pi * frequency / sampleRate
        let c = cos(w0), alpha = sin(w0) / (2 * q)
        return Biquad(b0: (1 + c) / 2, b1: -(1 + c), b2: (1 + c) / 2,
                      a0: 1 + alpha, a1: -2 * c, a2: 1 - alpha)
    }

    static func peaking(frequency: Double, gainDB: Double, q: Double, sampleRate: Double = 48_000) -> Biquad {
        let a = pow(10, gainDB / 40)
        let w0 = 2 * Double.pi * frequency / sampleRate
        let c = cos(w0), alpha = sin(w0) / (2 * q)
        return Biquad(b0: 1 + alpha * a, b1: -2 * c, b2: 1 - alpha * a,
                      a0: 1 + alpha / a, a1: -2 * c, a2: 1 - alpha / a)
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
    private var hpf: Biquad
    private var deMud: Biquad
    private var presence: Biquad
    private var air: Biquad
    private var rmsEnv = 0.0
    private var compGain = 1.0
    private var limEnv = 0.0

    // Speech-gated compressor constants (48 kHz)
    private let rmsAlpha = exp(-1.0 / (0.025 * 48_000))      // 25 ms RMS window
    private let compAttack = exp(-1.0 / (0.015 * 48_000))    // 15 ms attack
    private let compRelease = exp(-1.0 / (0.180 * 48_000))   // 180 ms release
    // -1 dBTP peak safety limiter constants (48 kHz)
    private let ceiling = 0.8912509381337456                 // 10^(-1 / 20)
    private let limAttack = exp(-1.0 / (0.0004 * 48_000))    // 0.4 ms fast peak catch
    private let limRelease = exp(-1.0 / (0.045 * 48_000))    // 45 ms transparent release

    init(enabled: Bool = true, limitPeaks: Bool = true, sampleRate: Double = 48_000) {
        self.enabled = enabled
        self.limitPeaks = limitPeaks
        self.hpf = .highPass(frequency: 75, sampleRate: sampleRate)
        self.deMud = .peaking(frequency: 280, gainDB: -1.5, q: 1.0, sampleRate: sampleRate)
        self.presence = .peaking(frequency: 3200, gainDB: 2.0, q: 0.9, sampleRate: sampleRate)
        self.air = .highShelf(frequency: 10_000, gainDB: 1.5, slope: 0.7, sampleRate: sampleRate)
    }

    mutating func reset() {
        hpf.reset()
        deMud.reset()
        presence.reset()
        air.reset()
        rmsEnv = 0
        compGain = 1
        limEnv = 0
    }

    /// Processes a 48 kHz frame in place. Peak protection runs when `limitPeaks` is on; EQ + speech-gated leveling run when `enabled`.
    mutating func process(_ samples: inout [Float]) {
        guard enabled || limitPeaks else { return }
        for i in samples.indices {
            var x = Double(samples[i])
            if !x.isFinite { x = 0 }

            if enabled {
                x = hpf.process(x)
                x = deMud.process(x)
                x = presence.process(x)
                x = air.process(x)

                // Speech-gated soft-knee compressor: active only above -45 dBFS so silence/pauses stay untouched
                rmsEnv = rmsAlpha * rmsEnv + (1 - rmsAlpha) * (x * x)
                let rms = sqrt(rmsEnv)
                let targetGain: Double
                if rms < 0.005623413 {  // < -45 dBFS: pause/noise floor -> unity gain (no pumping)
                    targetGain = 1.0
                } else {
                    let levelDB = 20 * log10(rms)
                    let thresholdDB = -22.0
                    let kneeDB = 6.0
                    let ratio = 2.2
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
                    // Gentle +2.5 dB makeup faded in smoothly from -45 to -33 dBFS
                    let gate = min(1.0, max(0.0, (levelDB + 45.0) / 12.0))
                    let gainDB = reductionDB + 2.5 * gate
                    targetGain = pow(10, gainDB / 20)
                }
                let coeff = targetGain < compGain ? compAttack : compRelease
                compGain = coeff * compGain + (1 - coeff) * targetGain
                x *= compGain
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
