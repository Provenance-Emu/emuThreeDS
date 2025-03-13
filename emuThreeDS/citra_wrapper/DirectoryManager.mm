//
//  DirectoryManager.mm
//  emuThreeDS
//
//  Created by Antique on 25/5/2023.
//

#include <Foundation/Foundation.h>
#import "DirectoryManager.h"

std::string DirectoryManager::DocumentDirectory() {
#if TARGET_OS_TV
    return [[[[[NSFileManager defaultManager] URLsForDirectory:NSCachesDirectory inDomains:NSUserDomainMask] firstObject] path] cStringUsingEncoding:NSUTF8StringEncoding];
#else
    return [[[[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject] path] cStringUsingEncoding:NSUTF8StringEncoding];
#endif
}
