def extract_info_from_url(url):
    # Check if the URL is a string
    if not isinstance(url, str):
        return None, None  # Use None for non-string to avoid processing errors
    parts = url.split('/')
    # Check if the split parts list is long enough
    if len(parts) > 9:
        season = parts[7]  # 'V' or 'W' for summer or winter
        demographic = parts[9]  # '1', '2', '3' for women, men, kids
        return season, demographic
    return None, None  # Return None if the URL structure is incomplete
