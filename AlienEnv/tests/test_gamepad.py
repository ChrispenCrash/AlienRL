import vgamepad as vg
from time import sleep

gamepad = vg.VX360Gamepad()

# gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)  # press the A button
# gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)  # press the left hat button

# gamepad.update()  # send the updated state to the computer

# # (...) A and left hat are pressed...

# gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)  # release the A button

gamepad.left_trigger_float(value_float=0.5)
gamepad.right_trigger_float(value_float=0.5)
gamepad.update()  # send the updated state to the computer
sleep(3)
gamepad.left_trigger_float(value_float=1.0)
gamepad.right_trigger_float(value_float=1.0)
gamepad.update()  # send the updated state to the computer
sleep(3)

gamepad.reset()  # reset the gamepad to default state
gamepad.update()  # send the updated state to the computer