use crate::controls;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;

pub async fn run(mut display: crate::window::Display) {
    let window = display.window.take().unwrap();
    let mut keyboard = controls::Keybind::new(VirtualKeyCode::Yen);
    let mut now = std::time::Instant::now();
    let mut frames = 0;

    display.event_loop.take().unwrap().run(move |event, _, control| {
        let controls = controls::Inputs::default();

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    log::warn!("Close request was send..");
                    *control = ControlFlow::Exit;
                }
                WindowEvent::ModifiersChanged(modi) => {
                    keyboard.modifier ^= modi;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    keyboard.key = input.virtual_keycode.unwrap();

                    if input.state == ElementState::Pressed {
                        if controls.matching_action(controls::Actions::Maximize, keyboard) {
                            if window.fullscreen().is_some() {
                                window.set_fullscreen(None);
                            } else {
                                let handle = window.current_monitor();
                                window.set_fullscreen(Some(Fullscreen::Borderless(handle)));
                            }
                        }

                        if controls.matching_action(controls::Actions::CloseRequest, keyboard) {
                            *control = ControlFlow::Exit;
                        }
                    }
                }
                WindowEvent::Resized(size) => {
                    let device = &display.device;
                    let max = crate::window::MIN_REAL_SIZE;

                    display.camera.resize(size);

                    display.surfaces.iter_mut().for_each(|surface| {
                        surface.config.width = size.width.max(max.width);
                        surface.config.height = size.height.max(max.height);
                        surface.handle.configure(device, &surface.config);
                    });
                }
                _ => (),
            },
            Event::RedrawRequested(_) => {

                frames += 1;
                if now.elapsed() >= std::time::Duration::from_secs(1) {
                    log::info!("frame rate: {frames}");

                    frames = 0;
                    now = std::time::Instant::now();
                }

                display.redraw();
                display.camera.update();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => (),
        }
    });
}
