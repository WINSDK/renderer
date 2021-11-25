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
    pub keymap: Vec<(Actions, Keybind)>,
}

impl Default for Inputs {
    fn default() -> Self {
        let mut keymap: Vec<(Actions, Keybind)> = Vec::new();

        keymap.push((
            Actions::Maximize,
            Keybind::new_with_modifier(
                VirtualKeyCode::F,
                ModifiersState::CTRL & ModifiersState::LOGO,
            ),
        ));

        keymap.push((
            Actions::CloseRequest,
            Keybind::new_with_modifier(VirtualKeyCode::Escape, ModifiersState::empty()),
        ));

        Self { speed: 100.0, sensitivity: 1.3, keymap }
    }
}

impl Inputs {
    #[allow(dead_code)]
    pub fn new(keymap: Vec<(Actions, Keybind)>) -> Self {
        Self { speed: 100.0, sensitivity: 1.3, keymap }
    }

    /// Iterates through the bound keys and returns whether an action has been executed.
    pub fn matching_action(&self, action: Actions, input: Keybind) -> bool {
        for (_, entry) in self.keymap.iter().filter(|(act, _)| act == &action) {
            if *entry == input {
                return true;
            }
        }

        false
    }

    pub fn insert(&mut self, action: Actions, keybind: Keybind) {
        self.keymap.push((action, keybind));
    }

    /// Handle keyboard input
    pub fn keyboard(&mut self, _key: KeyboardInput) {}

    /// Handle mouse input
    pub fn mouse(&mut self) {}
}
