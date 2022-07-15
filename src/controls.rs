use std::collections::HashMap;
use winit::event::{KeyboardInput, ModifiersState, VirtualKeyCode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keybind {
    pub modifier: ModifiersState,
    pub key: VirtualKeyCode,
}

impl Keybind {
    pub fn new(key: VirtualKeyCode) -> Self {
        Self { modifier: ModifiersState::empty(), key }
    }

    pub fn new_with_modifier(key: VirtualKeyCode, modifier: ModifiersState) -> Self {
        Self { modifier, key }
    }
}

#[allow(dead_code)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub enum Actions {
    Left,
    Right,
    Forward,
    Backward,
    Maximize,
    CloseRequest,
}

#[derive(Debug)]
pub struct Inputs {
    speed: f32,
    sensitivity: f32,
    pub keymap: HashMap<Actions, Vec<Keybind>>,
}

impl Default for Inputs {
    fn default() -> Self {
        let mut keymap = Inputs { speed: 100.0, sensitivity: 1.3, keymap: HashMap::new() };

        keymap.insert(
            Actions::Maximize,
            Keybind::new_with_modifier(
                VirtualKeyCode::F,
                ModifiersState::CTRL & ModifiersState::LOGO,
            ),
        );

        keymap.insert(
            Actions::CloseRequest,
            Keybind::new_with_modifier(VirtualKeyCode::Escape, ModifiersState::empty()),
        );

        keymap
    }
}

impl Inputs {
    #[allow(dead_code)]
    pub fn new(keymap: HashMap<Actions, Vec<Keybind>>) -> Self {
        Self { speed: 100.0, sensitivity: 1.3, keymap }
    }

    /// Iterates through the bound keys and returns whether an action has been executed.
    pub fn matching_action(&self, action: Actions, input: Keybind) -> bool {
        self.keymap.get(&action);

        if let Some(keybinds) = self.keymap.get(&action) {
            for keybind in keybinds {
                if keybind == &input {
                    return true;
                }
            }
        }

        false
    }

    pub fn insert(&mut self, action: Actions, keybind: Keybind) {
        if let Some(keybinds) = self.keymap.get_mut(&action) {
            keybinds.push(keybind);
        } else {
            self.keymap.insert(action, vec![keybind]);
        }
    }

    /// Handle keyboard input
    pub fn keyboard(&mut self, _key: KeyboardInput) {}

    /// Handle mouse input
    pub fn mouse(&mut self) {}
}
