//
//  EmulationViewController.swift
//  emuThreeDS
//
//  Created by Antique on 30/5/2023.
//

import AVFoundation
import AVFAudio
import Foundation
#if canImport(GameController)
import GameController
#endif
import MetalKit
import SwiftUI
import UIKit

class EmulationViewController : UIViewController {
    var emulationManager: EmulationManager
    
    var metalView = MTKView()
    var menuButton: UIButton!
    
    var virtualController = SuperController(elements: [GCInputLeftThumbstick, GCInputRightThumbstick, GCInputLeftShoulder, GCInputRightShoulder, GCInputLeftTrigger, GCInputRightTrigger, GCInputButtonA, GCInputButtonB, GCInputButtonX, GCInputButtonY])
    
    init(emulationManager: EmulationManager) {
        self.emulationManager = emulationManager
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        metalView.translatesAutoresizingMaskIntoConstraints = false
        metalView.device = MTLCreateSystemDefaultDevice()
        view.addSubview(metalView)
        view.addConstraints([
            metalView.topAnchor.constraint(equalTo: view.topAnchor),
            metalView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            metalView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            metalView.trailingAnchor.constraint(equalTo: view.trailingAnchor)
        ])
        
        var configuration = UIButton.Configuration.plain()
        configuration.image = UIImage(systemName: "gearshape")
        menuButton = UIButton(configuration: configuration)
        menuButton.translatesAutoresizingMaskIntoConstraints = false
        menuButton.menu = UIMenu(children: [
            UIMenu(title: "Controller", image: UIImage(systemName: "gamecontroller"), children: [
                UIAction(title: "Connect", handler: { _ in
                    self.connect()
                }),
                UIAction(title: "Disconnect", handler: { _ in
                    self.disconnect()
                }),
            ])
        ])
        menuButton.showsMenuAsPrimaryAction = true
        metalView.addSubview(menuButton)
        view.addConstraints([
            menuButton.topAnchor.constraint(equalTo: metalView.safeAreaLayoutGuide.topAnchor, constant: 10),
            menuButton.leadingAnchor.constraint(equalTo: metalView.safeAreaLayoutGuide.leadingAnchor, constant: 10)
        ])
        
        
        NotificationCenter.default.addObserver(self, selector: #selector(controllerDidBecomeCurrent), name: NSNotification.Name.GCControllerDidBecomeCurrent, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(audioRouteDidChange(_:)), name: AVAudioSession.routeChangeNotification, object: nil)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        if !emulationManager.hasConfigured {
            emulationManager.use(layer: metalView.layer as! CAMetalLayer)
        }
        
        emulationManager.run()
        connect()
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        disconnect()
    }
    
    
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        if emulationManager.hasConfigured, emulationManager.wrapper.isRunning {
            emulationManager.orientationChanged(orientation: UIDevice.current.orientation, with: metalView.layer as! CAMetalLayer)
        }
        disconnect(); connect()
    }
    
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)
        guard let touch = touches.first else {
            return
        }
        
        emulationManager.touchesBegan(at: touch.location(in: metalView))
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesMoved(touches, with: event)
        guard let touch = touches.first else {
            return
        }
        
        emulationManager.touchesMoved(at: touch.location(in: metalView))
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesEnded(touches, with: event)
        emulationManager.touchesEnded()
    }
    
    
    func connect() {
        Task {
            try await virtualController.connect()
        }
    }
    
    func disconnect() {
        virtualController.disconnect()
    }
    
    
    
    @objc func audioRouteDidChange(_ notification: Notification) {
        guard let reason = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? AVAudioSession.RouteChangeReason else {
            return
        }
        
        switch (reason) {
        case .newDeviceAvailable:
            do {
                try AVAudioSession.sharedInstance().setCategory(.playback, options: [.allowBluetooth, .allowBluetoothA2DP])
                try AVAudioSession.sharedInstance().setActive(true)
            } catch { print(error.localizedDescription) }
        case .oldDeviceUnavailable:
            do {
                try AVAudioSession.sharedInstance().setCategory(.playback)
                try AVAudioSession.sharedInstance().setActive(true)
            } catch { print(error.localizedDescription) }
        default: break
        }
    }
    
    @objc func controllerDidBecomeCurrent() {
        guard let currentController = GCController.current, let extendedGamepad = currentController.extendedGamepad else {
            return
        }
        
        extendedGamepad.buttonA.valueChangedHandler = EmulationInput.buttonA.valueChangedHandler
        extendedGamepad.buttonB.valueChangedHandler = EmulationInput.buttonB.valueChangedHandler
        extendedGamepad.buttonX.valueChangedHandler = EmulationInput.buttonX.valueChangedHandler
        extendedGamepad.buttonY.valueChangedHandler = EmulationInput.buttonY.valueChangedHandler
        
        extendedGamepad.leftShoulder.valueChangedHandler = EmulationInput.buttonL.valueChangedHandler
        extendedGamepad.rightShoulder.valueChangedHandler = EmulationInput.buttonR.valueChangedHandler
        
        extendedGamepad.leftTrigger.valueChangedHandler = EmulationInput.buttonZL.valueChangedHandler
        extendedGamepad.rightTrigger.valueChangedHandler = EmulationInput.buttonZR.valueChangedHandler
        
        extendedGamepad.buttonOptions?.valueChangedHandler = EmulationInput.buttonSelect.valueChangedHandler
        extendedGamepad.buttonMenu.valueChangedHandler = EmulationInput.buttonStart.valueChangedHandler
        
        extendedGamepad.leftThumbstick.valueChangedHandler = EmulationInput.circlePad.valueChangedHandler
        extendedGamepad.rightThumbstick.valueChangedHandler = EmulationInput.circlePadPro.valueChangedHandler
    }
}
