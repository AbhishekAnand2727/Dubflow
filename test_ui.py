from playwright.sync_api import sync_playwright
import time

def run_ui_test():
    print("Starting UI Test with Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # 1. Navigate to Home
        print("Navigating to http://localhost:5173...")
        page.goto("http://localhost:5173")
        page.wait_for_load_state("networkidle")
        page.screenshot(path="screenshot_home.png")
        print("[OK] Home page loaded. Screenshot saved.")

        # 2. Go to New Dub (Upload Page)
        print("Clicking 'New Dub'...")
        page.click("text=New Dub")
        page.wait_for_timeout(1000)
        page.screenshot(path="screenshot_upload.png")
        
        # 3. Verify Back Button
        if page.is_visible("text=Back to Dashboard"):
            print("[OK] 'Back to Dashboard' button is visible.")
        else:
            print("[FAIL] 'Back to Dashboard' button NOT found.")

        # 4. Click Back Button
        print("Clicking 'Back to Dashboard'...")
        page.click("text=Back to Dashboard")
        page.wait_for_timeout(1000)
        
        # Capture console logs
        page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))

        # 5. Open a Video Player
        print("Opening first video card...")
        page.wait_for_selector(".grid", timeout=5000)
        
        # Click the first card
        # We click the first child of the grid
        card = page.locator(".grid > div").first
        if card.is_visible():
            print("Found video card, clicking...")
            card.click()
        else:
            print("[FAIL] No video cards found in grid!")
        
        print("Waiting for player to load...")
        try:
            # Wait for video element
            page.wait_for_selector("video", timeout=10000)
            page.screenshot(path="screenshot_player.png")
            
            # 6. Verify Player Size
            video_container = page.locator("video").first
            if video_container.is_visible():
                box = video_container.bounding_box()
                print(f"[OK] Video player is visible. Size: {box['width']}x{box['height']}")
                
                if box['width'] > 600:
                    print("[OK] Video player size appears to be large (Width > 600px).")
                else:
                    print(f"[WARN] Video player might be small (Width: {box['width']}px).")
            else:
                print("[FAIL] Video player NOT found (is_visible=False).")
        except Exception as e:
            print(f"[FAIL] Video player NOT found (Timeout): {e}")
            page.screenshot(path="screenshot_player_fail.png")

        browser.close()
        print("UI Test Completed.")

if __name__ == "__main__":
    run_ui_test()
