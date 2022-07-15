use crate::controls;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;

pub async fn run(mut window: crate::window::Window) {
    let event_loop = window.get_event_loop();

    // YEN as a default key should not be the case.
    let mut keyboard = controls::Keybind::new(VirtualKeyCode::Yen);

    event_loop.run(move |event, _, control| {
        let window_handle = window.get_window_handle();
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
                            let window = window.get_window_handle();
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
                    let device = &window.display.device;
                    let max = crate::window::MIN_REAL_SIZE;

                    window.camera.resize(size);

                    window.surfaces.iter_mut().for_each(|surface| {
                        surface.config.width = size.width.max(max.width);
                        surface.config.height = size.height.max(max.height);
                        surface.handle.configure(device, &surface.config);
                    });
                }
                _ => (),
            },
            Event::RedrawRequested(_) => {
                window.redraw();
                window.camera.update();
            }
            Event::MainEventsCleared => {
                window_handle.request_redraw();
            }
            _ => (),
        }
    });
}
