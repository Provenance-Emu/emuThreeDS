//
//  CitraWrapper.h
//  emuThreeDS
//
//  Created by Antique on 22/5/2023.
//

#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN
@interface CitraWrapper : NSObject {
    CAMetalLayer *_metalLayer;
    NSString *_path;
    
    NSThread *_thread;
}

@property (nonatomic, readwrite) BOOL isPaused, isRunning;

+(CitraWrapper *) sharedInstance;
-(instancetype) init;


-(uint16_t*) GetIcon:(NSString *)path;
-(NSString *) GetPublisher:(NSString *)path;
-(NSString *) GetRegion:(NSString *)path;
-(NSString *) GetTitle:(NSString *)path;


-(void) useMetalLayer:(CAMetalLayer *)layer;
-(void) load:(NSString *)path;
-(void) pause;
-(void) run;
-(void) start;


-(void) touchesBegan:(CGPoint)point;
-(void) touchesMoved:(CGPoint)point;
-(void) touchesEnded;

#if !TARGET_OS_TV
-(void) orientationChanged:(UIDeviceOrientation)orientation with:(CAMetalLayer *)surface;
#endif
@end
NS_ASSUME_NONNULL_END
