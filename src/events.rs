use std::sync::Arc;
use winit::event::{ElementState, Event, ModifiersState, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;
use winit::window::Window as WindowHandle;

#[derive(Debug)]
struct State<'a> {
    control: &'a mut ControlFlow,
    modifier: Option<ModifiersState>,
    code: VirtualKeyCode,
}

impl<'a> State<'a> {
    pub fn handle_keyboard(&mut self, window: Arc<WindowHandle>) {
        match self.code {
            VirtualKeyCode::Escape => {
                *self.control = ControlFlow::Exit;
            }
            VirtualKeyCode::F11 => {
                if window.fullscreen().is_some() {
                    window.set_fullscreen(None);
                } else {
                    let handle = window.current_monitor();
                    window.set_fullscreen(Some(Fullscreen::Borderless(handle)));
                }
            }
            _ => (),
        }
    }
}

pub async fn run(mut window: crate::Window) {
    let event_loop = window.get_event_loop();

    event_loop.run(move |event, _, ref mut control| {
        let window_handle = window.get_window_handle();
        let mut state = State {
            control,
            modifier: None,
            code: VirtualKeyCode::F24,
        };

        *state.control = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    log::warn!("Close request was send..");
                    *state.control = ControlFlow::Exit;
                }
                WindowEvent::ModifiersChanged(m) => state.modifier = Some(m),
                WindowEvent::KeyboardInput { input, .. } => {
                    if input.state == ElementState::Pressed {
                        state.code = input.virtual_keycode.unwrap();
                        state.handle_keyboard(window.get_window_handle());
                    }
                }
                WindowEvent::Resized(size) => {
                    let display = &window.display;
                    window.swap_chains.iter_mut().for_each(|swap| {
                        let max = crate::MIN_REAL_SIZE;
                        swap.desc.width = size.width.max(max.width);
                        swap.desc.height = size.height.max(max.height);
                        swap.chain = display
                            .device
                            .create_swap_chain(&display.surface, &swap.desc);
                    });
                }
                _ => (),
            },
            Event::RedrawRequested(_) => window.redraw(),
            Event::MainEventsCleared => {
                window_handle.request_redraw();
            }
            _ => (),
        }
    });
}
