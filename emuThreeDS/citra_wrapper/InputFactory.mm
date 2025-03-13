//
//  InputFactory.mm
//  emuThreeDS
//
//  Created by Antique on 27/5/2023.
//

#import "InputBridge.h"
#import "InputFactory.h"

#include "emuThreeDS-Swift.h"
#include <thread>
#import <CoreMotion/CoreMotion.h>

@class EmulationInput;

std::unique_ptr<Input::AnalogDevice> AnalogFactory::Create(const Common::ParamPackage& params) {
    int button_id = params.Get("code", 0);
    AnalogInputBridge* emuInput = nullptr;
    switch ((Settings::NativeAnalog::Values)button_id) {
        case Settings::NativeAnalog::CirclePad:
            emuInput = EmulationInput.circlePad;
            break;
        case Settings::NativeAnalog::CStick:
            emuInput = EmulationInput.circlePadPro;
            break;
        case Settings::NativeAnalog::NumAnalogs:
            UNREACHABLE();
            break;
    }
    
    if (emuInput == nullptr)
        return {};
    return std::unique_ptr<AnalogBridge>([emuInput getCppBridge]);
}

std::unique_ptr<Input::ButtonDevice> ButtonFactory::Create(const Common::ParamPackage& params) {
    int button_id = params.Get("code", 0);
    ButtonInputBridge* emuInput = nullptr;
    switch ((Settings::NativeButton::Values)button_id) {
        case Settings::NativeButton::A:
            emuInput = EmulationInput.buttonA;
            break;
        case Settings::NativeButton::B:
            emuInput = EmulationInput.buttonB;
            break;
        case Settings::NativeButton::X:
            emuInput = EmulationInput.buttonX;
            break;
        case Settings::NativeButton::Y:
            emuInput = EmulationInput.buttonY;
            break;
        case Settings::NativeButton::Up:
            emuInput = EmulationInput.dpadUp;
            break;
        case Settings::NativeButton::Down:
            emuInput = EmulationInput.dpadDown;
            break;
        case Settings::NativeButton::Left:
            emuInput = EmulationInput.dpadLeft;
            break;
        case Settings::NativeButton::Right:
            emuInput = EmulationInput.dpadRight;
            break;
        case Settings::NativeButton::L:
            emuInput = EmulationInput.buttonL;
            break;
        case Settings::NativeButton::R:
            emuInput = EmulationInput.buttonR;
            break;
        case Settings::NativeButton::Start:
            emuInput = EmulationInput.buttonStart;
            break;
        case Settings::NativeButton::Select:
            emuInput = EmulationInput.buttonSelect;
            break;
        case Settings::NativeButton::ZL:
            emuInput = EmulationInput.buttonZL;
            break;
        case Settings::NativeButton::ZR:
            emuInput = EmulationInput.buttonZR;
            break;
        case Settings::NativeButton::Debug:
        case Settings::NativeButton::Gpio14:
        case Settings::NativeButton::Home:
        case Settings::NativeButton::NumButtons:
            emuInput = EmulationInput._buttonDummy;
    }
    
    if (emuInput == nullptr)
        return {};
    return std::unique_ptr<ButtonBridge<bool>>([emuInput getCppBridge]);
}

static std::shared_ptr<MotionFactory> motion;

namespace {
using Common::Vec3;
}

class Motion : public Input::MotionDevice {
    std::chrono::microseconds update_period;
    
    mutable std::atomic<Vec3<float>> acceleration{};
    mutable std::atomic<Vec3<float>> rotation{};
    static_assert(decltype(acceleration)::is_always_lock_free, "vectors are not lock free");
    std::thread poll_thread;
    std::atomic<bool> stop_polling = false;
    
    CMMotionManager *motionManager;
    
    static Vec3<float> TransformAxes(Vec3<float> in) {
        // 3DS   Y+            Phone     Z+
        // on    |             laying    |
        // table |             in        |
        //       |_______ X-   portrait  |_______ X+
        //      /              mode     /
        //     /                       /
        //    Z-                      Y-
        Vec3<float> out;
        out.y = in.z;
        // rotations are 90 degrees counter-clockwise from portrait
        switch (screen_rotation) {
            case 0:
                out.x = -in.x;
                out.z = in.y;
                break;
            case 1:
                out.x = in.y;
                out.z = in.x;
                break;
            case 2:
                out.x = in.x;
                out.z = -in.y;
                break;
            case 3:
                out.x = -in.y;
                out.z = -in.x;
                break;
            default:
                UNREACHABLE();
        }
        return out;
    }
    
public:
    Motion(std::chrono::microseconds update_period_, bool asynchronous = false)
    : update_period(update_period_) {
        if (asynchronous) {
            poll_thread = std::thread([this] {
                Construct();
                auto start = std::chrono::high_resolution_clock::now();
                while (!stop_polling) {
                    Update();
                    std::this_thread::sleep_until(start += update_period);
                }
                Destruct();
            });
        } else {
            Construct();
        }
    }
    
    std::tuple<Vec3<float>, Vec3<float>> GetStatus() const override {
        if (std::thread::id{} == poll_thread.get_id()) {
            Update();
        }
        return {acceleration, rotation};
    }
    
    void Construct() {
        motionManager = [[CMMotionManager alloc] init];
        EnableSensors();
    }
    
    void Destruct() {
        
    }
    
    void Update() const {
        CMDeviceMotion *motion = [motionManager deviceMotion];
        
        acceleration = {
            (motion.gravity.x + motion.userAcceleration.x),
            (motion.gravity.y + motion.userAcceleration.y),
            (motion.gravity.z + motion.userAcceleration.z)
        };
        
        rotation = {
            motion.rotationRate.x,
            motion.rotationRate.y,
            motion.rotationRate.z
        };
    }
    
    void DisableSensors() {
        [motionManager stopDeviceMotionUpdates];
    }
    
    void EnableSensors() {
        [motionManager startDeviceMotionUpdates];
    }
};

std::unique_ptr<Input::MotionDevice> MotionFactory::Create(const Common::ParamPackage &params) {
    std::chrono::milliseconds update_period{params.Get("update_period", 4)};
    std::unique_ptr<Motion> motion = std::make_unique<Motion>(update_period);
    _motion = motion.get();
    return std::move(motion);
}

void MotionFactory::EnableSensors() {
    _motion->EnableSensors();
};

void MotionFactory::DisableSensors() {
    _motion->DisableSensors();
};


MotionFactory* MotionHandler() {
    return motion.get();
}



