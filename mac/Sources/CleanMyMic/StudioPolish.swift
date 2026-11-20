// Lightweight broadcast finishing chain after the neural model (48 kHz mono, zero extra latency):
// 1. 75 Hz Butterworth high-pass (removes desk thumps, plosives, DC)
// 2. Subtle studio vocal EQ (gentle 280 Hz de-mud, 3.2 kHz presence, 10 kHz air) when `studioPolish` is on
// 3. Speech-gated soft-knee leveler/compressor (never boosts silence/pauses) when `studioPolish` is on
// 4. Always-on -1 dBTP peak safety limiter (prevents Core ML overshoots from clipping Core Audio)
import Foundation

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
