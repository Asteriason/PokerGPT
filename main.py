import os
import openai
import pygetwindow as gw
from colorama import init
import Quartz
import os
import multiprocessing

os.environ["SDL_VIDEODRIVER"] = "quartz"
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Fix macOS forking issues


from game_state             import GameState
from gui                    import GUI
from hero_action            import HeroAction
from poker_assistant        import PokerAssistant
from audio_player           import AudioPlayer
from read_poker_table       import ReadPokerTable
from hero_hand_range        import PokerHandRangeDetector
from hero_info              import HeroInfo

def main():

    # Ask the user for the hero player number ( 1- 6 , starting from bottom(1))
    while True:
        try:
            hero_player_number = int(input("Enter hero player number (1-6): "))
            if 1 <= hero_player_number <= 6:
                break
            else:
                print("Invalid number. Please enter a number between 1 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")


    api_key                 = os.getenv('OPENAI_API_KEY')
    openai_client           = openai.OpenAI(api_key=api_key)
    poker_window            = locate_poker_window()
    init(autoreset=True)

    # Initialize all the instances
    
    if poker_window is not None:

        audio_player            = AudioPlayer( openai_client )
        hero_action             = HeroAction( poker_window )
        
        hero_info               = HeroInfo()
        hero_hand_range         = PokerHandRangeDetector()

        game_state              = GameState( hero_action, audio_player )
        poker_assistant         = PokerAssistant( openai_client, hero_info, game_state, hero_action, audio_player )
       
        gui                     = GUI( game_state, poker_assistant )
        read_poker_table        = ReadPokerTable( poker_window, hero_info, hero_hand_range, hero_action, poker_assistant, game_state )

        

        setup_read_poker_table( read_poker_table=read_poker_table )

        # Update hero player number in game state
        game_state.update_player(hero_player_number, hero=True)

        game_state.hero_player_number = hero_player_number

        game_state.extract_blinds_from_title()

        # Start the GUI
        gui.run()


def locate_poker_window():
    """Locate the poker client window on macOS."""
    
    options = Quartz.kCGWindowListOptionOnScreenOnly
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

    poker_window = None

    for window in window_list:
        owner_name = window.get("kCGWindowOwnerName", "")
        window_bounds = window.get("kCGWindowBounds", {})

        if owner_name == "PokerStars" and "Height" in window_bounds and "Width" in window_bounds:
            print(f"✅ Found Poker Table: {window}")

            poker_window = {
                "width": window_bounds["Width"],
                "height": window_bounds["Height"],
                "x": window_bounds["X"],
                "y": window_bounds["Y"]
            }
            break

    if poker_window:
        return poker_window
    else:
        print("❌ No Poker Table found. Make sure the game is open and visible.")
        return None



def resize_poker_window( window, width, height ):
    """Resize the poker client window to the specified width and height."""

    window.resizeTo(width, height)
    print(f"Resized window to: Width={width}, Height={height}")
        
 

def setup_read_poker_table(read_poker_table):

    # Start continuous detection of the poker table
    read_poker_table.start_continuous_detection()



if __name__ == "__main__":
    main()

# Run script:
# python main.py