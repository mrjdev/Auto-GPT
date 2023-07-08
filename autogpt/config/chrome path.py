import subprocess

def find_chrome_path():
    try:
        chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        return chrome_path
    except Exception as e:
        print(f"Error finding Chrome path: {e}")
        return None

# Call the function to get the Chrome path
chrome_path = find_chrome_path()
if chrome_path:
    print("Google Chrome path:", chrome_path)
else:
    print("Failed to find Google Chrome path.")
