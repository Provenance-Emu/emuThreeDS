//
//  file_handle.cpp
//  emuThreeDS
//
//  Created by Antique on 27/5/2023.
//

#include <Foundation/Foundation.h>
#include <UIKit/UIKit.h>
#include "file_handle.h"

float ResolutionHandle::StatusBarHeight() {
#if TARGET_OS_TV
    return 0;
#else
    UIWindowScene *windowScene = (UIWindowScene *)[[[[UIApplication sharedApplication] connectedScenes] allObjects] firstObject];
    return [[windowScene statusBarManager] statusBarFrame].size.height;
#endif
}

float ResolutionHandle::GetScreenWidth() {
    return [[UIScreen mainScreen] nativeBounds].size.width;
}

float ResolutionHandle::GetScreenHeight() {
    return [[UIScreen mainScreen] nativeBounds].size.height;
}

float ResolutionHandle::CombinedInsets() {
    UIWindowScene* windowScene = (UIWindowScene *)[[UIApplication sharedApplication] connectedScenes].allObjects.firstObject;
    UIWindow* window = (UIWindow *)windowScene.windows.firstObject;
    return window.safeAreaInsets.top + window.safeAreaInsets.bottom;
}


bool ResolutionHandle::IsPortrait() {
#if TARGET_OS_TV
    return true;
#else
    __block UIWindowScene *scene;
    dispatch_async(dispatch_get_main_queue(), ^{
        scene = (UIWindowScene *)[[UIApplication sharedApplication] connectedScenes].allObjects.lastObject;
    });
    
    return scene.interfaceOrientation == UIInterfaceOrientationPortrait;
#endif
}



bool DefaultsHandle::BoolForKey(std::string key) {
    return [[NSUserDefaults standardUserDefaults] boolForKey:[NSString stringWithUTF8String:key.c_str()]];
}
