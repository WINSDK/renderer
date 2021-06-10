use winit::event_loop::ControlFlow;
use winit::event::{Event, WindowEvent};

pub fn run(window: crate::Window) {
    //let event_loop = window.display.event_loop;

    //event_loop.run(move |event, _, control_flow| {
    //    *control_flow = ControlFlow::Poll;

    //    match event {
    //        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
    //            log::warn!("Close request was send..");
    //            *control_flow = ControlFlow::Exit;
    //        }
    //        Event::MainEventsCleared => {
    //            window.display.window.request_redraw();
    //        }
    //        _ => (),
    //    }
    //});
}
