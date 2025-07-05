import time
import os
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv

load_dotenv()

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10

# Azure OpenAI Configuration
azure_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview"
)
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Initialize embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

def get_embedding(text: str):
    return model.encode([text])[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def escape_xpath_string(text):
    """Properly escape strings for XPath"""
    if "'" not in text:
        return f"'{text}'"
    elif '"' not in text:
        return f'"{text}"'
    else:
        # Handle strings with both single and double quotes
        parts = text.split("'")
        return "concat('" + "', \"'\", '".join(parts) + "')"

def generate_robust_xpaths(tag, attrs, text, soup_element):
    """Generate multiple XPath strategies for better element location"""
    xpaths = []
    
    # Clean text for XPath usage
    clean_text = text.strip()[:50] if text else ""
    
    # Strategy 1: ID (most reliable)
    if attrs.get('id'):
        xpaths.append(f"//{tag}[@id='{attrs['id']}']")
    
    # Strategy 2: Class combinations
    if attrs.get('class'):
        classes = attrs['class']
        if isinstance(classes, list):
            # Try exact class match
            class_str = ' '.join(classes)
            xpaths.append(f"//{tag}[@class='{class_str}']")
            
            # Try individual class contains
            for cls in classes[:2]:  # Limit to first 2 classes
                if cls and len(cls) > 2:
                    xpaths.append(f"//{tag}[contains(@class, '{cls}')]")
    
    # Strategy 3: Text content (for elements with short, unique text)
    if clean_text and len(clean_text) < 30:
        # Exact text match
        xpaths.append(f"//{tag}[text()={escape_xpath_string(clean_text)}]")
        # Partial text match
        if len(clean_text) > 5:
            xpaths.append(f"//{tag}[contains(text(), {escape_xpath_string(clean_text[:20])})]")
    
    # Strategy 4: Attribute combinations
    useful_attrs = ['name', 'type', 'role', 'data-testid', 'aria-label']
    for attr in useful_attrs:
        if attrs.get(attr):
            xpaths.append(f"//{tag}[@{attr}='{attrs[attr]}']")
    
    # Strategy 5: Parent-child relationships for unique positioning
    if soup_element.parent:
        parent = soup_element.parent
        if parent.get('id'):
            xpaths.append(f"//*[@id='{parent.get('id')}']//{tag}")
        elif parent.get('class'):
            parent_classes = parent.get('class', [])
            if parent_classes:
                parent_class = parent_classes[0]
                xpaths.append(f"//*[contains(@class, '{parent_class}')]//{tag}")
    
    # Strategy 6: Position-based (last resort)
    if not xpaths:
        siblings = soup_element.parent.find_all(tag) if soup_element.parent else []
        if len(siblings) > 1:
            try:
                position = siblings.index(soup_element) + 1
                xpaths.append(f"({tag})[{position}]")
            except ValueError:
                pass
    
    return xpaths[:5]  # Return top 5 strategies

def test_xpath_with_selenium(driver, xpaths):
    """Test XPaths with Selenium and return the first working one"""
    wait = WebDriverWait(driver, 2)
    
    for xpath in xpaths:
        try:
            element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            return xpath, element
        except (TimeoutException, NoSuchElementException):
            continue
    
    return None, None

def get_interactive_elements(soup, driver):
    """Extract interactive elements with robust XPath generation"""
    INTERACTIVE_TAGS = ['a', 'button', 'input', 'select', 'textarea', 'div']
    elements = []
    
    for tag in soup.find_all(INTERACTIVE_TAGS):
        text = tag.get_text(strip=True)
        
        # Skip elements without meaningful content
        if not text or len(text) < 2:
            continue
        
        # Skip hidden elements (basic check)
        if tag.get('style') and 'display:none' in tag.get('style').replace(' ', ''):
            continue
            
        # Generate multiple XPath strategies
        xpaths = generate_robust_xpaths(tag.name, tag.attrs, text, tag)
        
        # Test XPaths with Selenium to find working one
        working_xpath, selenium_element = test_xpath_with_selenium(driver, xpaths)
        
        if working_xpath and selenium_element:
            # Verify element is actually clickable
            try:
                if selenium_element.is_displayed() and selenium_element.is_enabled():
                    elements.append({
                        "tag": tag.name,
                        "text": text,
                        "attrs": tag.attrs,
                        "xpath": working_xpath,
                        "selenium_element": selenium_element,
                        "all_xpaths": xpaths  # Keep alternatives for debugging
                    })
            except Exception as e:
                print(f"Element validation failed: {e}")
                continue
    
    return elements

def describe_element(elem):
    """Create enhanced description with action context"""
    action_map = {
        'a': "link to ",
        'button': "button for ",
        'input': "input field for ",
        'select': "dropdown for ",
        'div': "interactive element for "
    }
    # action_hint = action_map.get(elem["tag"], "")
    return f"{elem['text']}"

def llm_select_action_azure(query, candidates):
    """Use Azure OpenAI GPT-4o to select best action"""
    prompt = f"""
    Web interaction task: {query}
    Choose the BEST element to interact with from these candidates:
    
    {chr(10).join([f"{i+1}. {desc}" for i, desc in enumerate(candidates)])}
    
    Respond ONLY with the number of your choice (1-{len(candidates)}).
    """
    
    try:
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You're a web interaction assistant. Select the most relevant UI element for the given task."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2
        )
        return int(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Azure OpenAI error: {e}")
        return 1

def is_element_clickable_by_attributes(soup_element):
    """Check if element has clickable attributes"""
    clickable_attrs = ['onclick', 'href', 'role']
    clickable_roles = ['button', 'link', 'tab', 'menuitem']
    
    # Direct clickable attributes
    if any(soup_element.get(attr) for attr in clickable_attrs):
        return True
    
    # Clickable roles
    if soup_element.get('role') in clickable_roles:
        return True
    
    # Input elements that are clickable
    if soup_element.name == 'input':
        input_type = soup_element.get('type', '').lower()
        if input_type in ['button', 'submit', 'checkbox', 'radio']:
            return True
    
    return False

def has_click_handler_style(soup_element):
    """Check if element has CSS that suggests it's clickable"""
    style = soup_element.get('style', '')
    class_names = ' '.join(soup_element.get('class', []))
    
    # Look for cursor pointer in inline styles
    if 'cursor:pointer' in style.replace(' ', '') or 'cursor: pointer' in style:
        return True
    
    # Common clickable class patterns
    clickable_patterns = [
        r'clickable', r'click', r'btn', r'button', r'link', 
        r'interactive', r'hover', r'pointer'
    ]
    
    for pattern in clickable_patterns:
        if re.search(pattern, class_names, re.IGNORECASE):
            return True
    
    return False

def validate_clickable_with_selenium(driver, xpath):
    """Validate that element is actually clickable in Selenium"""
    try:
        wait = WebDriverWait(driver, 2)
        element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        return element if element.is_displayed() and element.is_enabled() else None
    except (TimeoutException, NoSuchElementException):
        return None

def generate_xpath_for_element(soup_element):
    """Generate XPath for a BeautifulSoup element"""
    # Try ID first (most reliable)
    if soup_element.get('id'):
        return f"//{soup_element.name}[@id='{soup_element.get('id')}']"
    
    # Try data-testid
    if soup_element.get('data-testid'):
        return f"//{soup_element.name}[@data-testid='{soup_element.get('data-testid')}']"
    
    # Try class combination
    if soup_element.get('class'):
        classes = soup_element.get('class')
        if len(classes) == 1:
            return f"//{soup_element.name}[@class='{classes[0]}']"
        else:
            # Use contains for multiple classes
            return f"//{soup_element.name}[contains(@class, '{classes[0]}')]"
    
    # Try href for links
    if soup_element.get('href'):
        return f"//a[@href='{soup_element.get('href')}']"
    
    # Fallback to text content (risky but sometimes necessary)
    text = soup_element.get_text(strip=True)
    if text and len(text) < 50:
        return f"//{soup_element.name}[contains(text(), '{text[:30]}')]"
    
    return None

def find_clickable_element_hybrid(content_div_soup, driver, max_levels=3):
    """
    Hybrid approach to find the actual clickable element for a content div
    
    Args:
        content_div_soup: BeautifulSoup element (the div with content)
        driver: Selenium WebDriver instance
        max_levels: Maximum levels to traverse upward
    
    Returns:
        dict with 'element' (selenium element), 'xpath', 'strategy_used', 'soup_element'
        or None if no clickable element found
    """
    
    # Strategy 1: Check if the div itself is clickable
    if content_div_soup.name in ['a', 'button'] or is_element_clickable_by_attributes(content_div_soup):
        xpath = generate_xpath_for_element(content_div_soup)
        if xpath:
            selenium_element = validate_clickable_with_selenium(driver, xpath)
            if selenium_element:
                return {
                    'element': selenium_element,
                    'xpath': xpath,
                    'strategy_used': 'direct_element',
                    'soup_element': content_div_soup
                }
    
    # Strategy 2: Check immediate parent
    current_element = content_div_soup.parent
    level = 1
    
    while current_element and level <= max_levels:
        # Skip non-tag elements (like NavigableString)
        if not hasattr(current_element, 'name') or not current_element.name:
            current_element = current_element.parent
            continue
        
        # Check if current element is clickable by tag name
        if current_element.name in ['a', 'button']:
            xpath = generate_xpath_for_element(current_element)
            if xpath:
                selenium_element = validate_clickable_with_selenium(driver, xpath)
                if selenium_element:
                    return {
                        'element': selenium_element,
                        'xpath': xpath,
                        'strategy_used': f'parent_level_{level}_tag',
                        'soup_element': current_element
                    }
        
        # Strategy 3: Check if parent has clickable attributes
        if is_element_clickable_by_attributes(current_element):
            xpath = generate_xpath_for_element(current_element)
            if xpath:
                selenium_element = validate_clickable_with_selenium(driver, xpath)
                if selenium_element:
                    return {
                        'element': selenium_element,
                        'xpath': xpath,
                        'strategy_used': f'parent_level_{level}_attributes',
                        'soup_element': current_element
                    }
        
        # Strategy 4: Check CSS/class-based clickability
        if has_click_handler_style(current_element):
            xpath = generate_xpath_for_element(current_element)
            if xpath:
                selenium_element = validate_clickable_with_selenium(driver, xpath)
                if selenium_element:
                    return {
                        'element': selenium_element,
                        'xpath': xpath,
                        'strategy_used': f'parent_level_{level}_style',
                        'soup_element': current_element
                    }
        
        # Move to next parent
        current_element = current_element.parent
        level += 1
    
    # Strategy 5: Look for nearest ancestor with href or onclick (broader search)
    current_element = content_div_soup.parent
    level = 1
    
    while current_element and level <= max_levels * 2:  # Search a bit further
        if not hasattr(current_element, 'name') or not current_element.name:
            current_element = current_element.parent
            continue
        
        # Look for any element with href or onclick attributes
        if current_element.get('href') or current_element.get('onclick'):
            xpath = generate_xpath_for_element(current_element)
            if xpath:
                selenium_element = validate_clickable_with_selenium(driver, xpath)
                if selenium_element:
                    return {
                        'element': selenium_element,
                        'xpath': xpath,
                        'strategy_used': f'ancestor_level_{level}_handlers',
                        'soup_element': current_element
                    }
        
        current_element = current_element.parent
        level += 1
    
    # Strategy 6: LLM-assisted analysis (placeholder for future implementation)
    # llm_result = analyze_with_llm(content_div_soup, driver)
    # if llm_result:
    #     return llm_result
    
    return None

def analyze_with_llm(content_div_soup, driver):
    """
    Placeholder for LLM-assisted clickable element detection
    
    Args:
        content_div_soup: BeautifulSoup element with the content
        driver: Selenium WebDriver instance
    
    Returns:
        dict with clickable element info or None
    """
    # TODO: Implement LLM analysis
    # 1. Extract HTML context around the content_div_soup (maybe 3-5 levels up)
    # 2. Send to LLM with prompt asking for the best clickable element
    # 3. Parse LLM response and validate with Selenium
    # 4. Return result in same format as other strategies
    
    print("LLM analysis not implemented yet")
    return None

# Example usage function
def get_interactive_elements_with_hybrid_approach(soup, driver):
    """
    Enhanced version of get_interactive_elements using hybrid clickable detection
    """
    # Find all divs with content (your original logic)
    content_divs = []
    for div in soup.find_all(['div', 'span', 'p']):  # Expanded to include more content containers
        text = div.get_text(strip=True)
        if text and len(text) >= 2:
            content_divs.append(div)
    
    interactive_elements = []
    
    for content_div in content_divs:
        clickable_result = find_clickable_element_hybrid(content_div, driver)
        
        if clickable_result:
            interactive_elements.append({
                'text': content_div.get_text(strip=True),
                'xpath': clickable_result['xpath'],
                'selenium_element': clickable_result['element'],
                'strategy_used': clickable_result['strategy_used'],
                'clickable_soup_element': clickable_result['soup_element'],
                'content_soup_element': content_div
            })
            
            print(f"Found clickable element using strategy: {clickable_result['strategy_used']}")
            print(f"Content: {content_div.get_text(strip=True)[:50]}...")
            print(f"XPath: {clickable_result['xpath']}")
            print("---")
    
    return interactive_elements

def perform_web_action(url, query_text):
    """Complete workflow: analyze page and perform action with robust XPath handling"""
    # Setup browser
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        
        # Wait for page to load completely
        wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
        time.sleep(2)
        
        # Parse DOM
        soup = BeautifulSoup(driver.page_source, "html.parser")
        elements = get_interactive_elements_with_hybrid_approach(soup, driver)
        
        print(f"Found {len(elements)} clickable elements")
        # for e in elements:
        #     print(f"Text: {e['text']} - XPath: {e['xpath']}")
        
        # if not elements:
        #     print("No interactive elements found!")
        #     return
        
        # Generate embeddings
        query_embedding = get_embedding(query_text)
        descriptions = [describe_element(e) for e in elements]
        element_embeddings = model.encode(descriptions)
        
        # Rank elements
        scores = [cosine_similarity(query_embedding, emb) for emb in element_embeddings]
        ranked = sorted(
            zip(scores, elements, descriptions),
            key=lambda x: x[0],
            reverse=True
        )[:TOP_K]
        
        # print("\nTop candidates:")
        # for i, (score, elem, desc) in enumerate(ranked):
        #     print(f"{i+1}. [{score:.3f}] {elem}")
        
        # GPT-4o selection
        candidate_descs = [desc for _, _, desc in ranked]
        best_index = llm_select_action_azure(query_text, candidate_descs) - 1
        
        # Execute action
        best_element = ranked[best_index][1]
        print(f"\nSelected action: Clicking element - {best_element['text']}")
        print(f"Using XPath: {best_element['xpath']}")
        print(f"Using selenium elmeent: {best_element['selenium_element']}")
        
        # Perform the click using the pre-validated Selenium element
        try:
            selenium_element = best_element['selenium_element']
            
            # Scroll element into view
            driver.execute_script("arguments[0].scrollIntoView(true);", selenium_element)
            time.sleep(0.5)
            
            # Try click
            selenium_element.click()
            print("✅ Click successful!")
            
            # Validate action
            time.sleep(4)
            new_url = driver.current_url
            print(f"New URL: {new_url}")
            
            if new_url != url:
                print("✅ Page navigation detected - action likely successful")
                # time.sleep(2)
                driver.save_screenshot("action_success.png")
            else:
                print("ℹ️ No navigation detected - check if action was performed")
                
        except Exception as click_error:
            print(f"❌ Click failed: {click_error}")
            
            # Try alternative XPaths
            print("Trying alternative XPaths...")
            for alt_xpath in best_element.get('all_xpaths', []):
                if alt_xpath != best_element['xpath']:
                    try:
                        alt_element = driver.find_element(By.XPATH, alt_xpath)
                        driver.execute_script("arguments[0].scrollIntoView(true);", alt_element)
                        time.sleep(0.5)
                        alt_element.click()
                        print(f"✅ Alternative XPath worked: {alt_xpath}")
                        break
                    except Exception:
                        continue
            else:
                print("❌ All XPath alternatives failed")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    url = "https://demo.saleor.io/default-channel/products"
    query = "Add saleor balance 420 shoes to the cart"
    perform_web_action(url, query)