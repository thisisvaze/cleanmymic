#import "ObjCTry.h"

NSError *objcTry(void (NS_NOESCAPE ^block)(void)) {
    @try {
        block();
        return nil;
    } @catch (NSException *e) {
        return [NSError errorWithDomain:e.name code:0 userInfo:@{NSLocalizedDescriptionKey: e.reason ?: e.name}];
    }
}
