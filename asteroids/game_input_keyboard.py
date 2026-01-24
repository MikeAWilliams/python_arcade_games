import arcade
from game import InputMethod, Action


class KeyboardInput(InputMethod):
    """Keyboard-based input implementation"""

    def __init__(self):
        self.current_action = Action.NO_ACTION
        self.shoot_requested = False

    def get_move(self) -> Action:
        """Returns the current action based on keyboard state"""
        # Shoot is a one-time action, so we clear it after returning
        if self.shoot_requested:
            self.shoot_requested = False
            return Action.SHOOT

        return self.current_action

    def on_key_press(self, key, modifiers):
        """Handle keyboard press events - sets internal state"""
        if key == arcade.key.LEFT:
            self.current_action = Action.TURN_LEFT
        elif key == arcade.key.RIGHT:
            self.current_action = Action.TURN_RIGHT
        elif key == arcade.key.UP:
            self.current_action = Action.DECELERATE
        elif key == arcade.key.DOWN:
            self.current_action = Action.ACCELERATE
        elif key == arcade.key.SPACE:
            self.shoot_requested = True

    def on_key_release(self, key, modifiers):
        """Handle keyboard release events - resets to NO_ACTION"""
        if key == arcade.key.LEFT and self.current_action == Action.TURN_LEFT:
            self.current_action = Action.NO_ACTION
        elif key == arcade.key.RIGHT and self.current_action == Action.TURN_RIGHT:
            self.current_action = Action.NO_ACTION
        elif key == arcade.key.UP and self.current_action == Action.DECELERATE:
            self.current_action = Action.NO_ACTION
        elif key == arcade.key.DOWN and self.current_action == Action.ACCELERATE:
            self.current_action = Action.NO_ACTION
