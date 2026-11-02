#import <Foundation/Foundation.h>

/// Runs `block`; an Objective-C exception it raises comes back as an NSError instead of killing the app.
/// AVAudioEngine raises them (connect with a format the device just changed under us) and Swift can't catch them.
NSError * _Nullable objcTry(void (NS_NOESCAPE ^ _Nonnull block)(void));
