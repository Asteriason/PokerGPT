import Quartz
import cv2
import numpy as np
import pytesseract


def preprocess_hero_cards(hero_img):
    """Apply preprocessing to extract clear text from hero cards."""
    gray = cv2.cvtColor(hero_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)  # Binarization

    return thresh

def extract_text_from_image(img):
    """Use Tesseract OCR to extract text from an image."""
    config = "--psm 6 -c tessedit_char_whitelist=23456789TJQKA"  # Only allow valid card values
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()


def get_poker_window():
    """Finds the PokerStars game table window and returns its ID and bounds (x, y, width, height)."""
    options = Quartz.kCGWindowListOptionOnScreenOnly
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

    for window in window_list:
        owner_name = window.get("kCGWindowOwnerName", "")
        window_title = window.get("kCGWindowName", "")

        if "PokerStars" in owner_name and "No Limit Hold'em" in window_title:
            bounds = window["kCGWindowBounds"]
            x, y = int(bounds["X"]), int(bounds["Y"])
            width, height = int(bounds["Width"]), int(bounds["Height"])
            window_id = window["kCGWindowNumber"]
            print(f"✅ Found Poker Table Window: ID {window_id}, {x}, {y}, {width}, {height}")
            return window_id, x, y, width, height

    print("❌ PokerStars table window not found!")
    return None

def capture_window(window_id, x, y, width, height):
    """Captures the PokerStars table using its window ID."""
    image_ref = Quartz.CGWindowListCreateImage(
        Quartz.CGRectMake(x, y, width, height),
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
        Quartz.kCGWindowImageDefault
    )

    if not image_ref:
        print("❌ Failed to capture window image.")
        return None

    width = Quartz.CGImageGetWidth(image_ref)
    height = Quartz.CGImageGetHeight(image_ref)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)

    provider = Quartz.CGImageGetDataProvider(image_ref)
    image_data = Quartz.CGDataProviderCopyData(provider)

    np_buffer = np.frombuffer(image_data, dtype=np.uint8)

    if np_buffer.size == 0:
        print("❌ Error: Image data buffer is empty. No valid pixels.")
        return None

    np_buffer = np_buffer.reshape((height, bytes_per_row // 4, 4))[:, :width, :]

    return cv2.cvtColor(np_buffer, cv2.COLOR_RGBA2RGB)

def extract_cards(img):
    """Extract hero and community cards from the captured image dynamically."""
    height, width, _ = img.shape
    
    # 🌎 Community Cards (Middle Center)
    comm_y1, comm_x1 = height // 3, width // 4
    comm_y2, comm_x2 = height // 3 + 100, width * 3 // 4
    community_cards_region = img[comm_y1:comm_y2, comm_x1:comm_x2]

    # 🃏 Hero Cards - Placed at 65% of the way from the bottom to the community cards
    hero_y1 = int((comm_y2 + height) // 2.1)  # 60-65% down from the community cards
    hero_y2 = hero_y1 + 100
    hero_x1, hero_x2 = width // 2 - 100, width // 2 + 100
    hero_card_region = img[hero_y1:hero_y2, hero_x1:hero_x2]
    
    # Preprocess hero cards for better OCR
    processed_hero_cards = preprocess_hero_cards(hero_card_region)

    # Extract text
    hero_text = extract_text_from_image(processed_hero_cards)

    print(f"🃏 Detected Hero Cards: {hero_text}")

    # Show the cropped images
    cv2.imshow("Hero Cards", hero_card_region)
    cv2.imshow("Community Cards", community_cards_region)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return hero_card_region, community_cards_region


# ✅ Run the capture process
poker_window = get_poker_window()
if poker_window:
    window_id, x, y, width, height = poker_window
    img = capture_window(window_id, x, y, width, height)

    if img is not None:
        cv2.imshow("Captured Poker Table", img)
        hero_cards, community_cards = extract_cards(img)
