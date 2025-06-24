def nchoosek(n: int, k: int) -> int:
    """Calculate the binomial coefficient n choose k.
    
    Args:
        n: Total number of items
        k: Number of items to choose
        
    Returns:
        The binomial coefficient (n k)
    """
    if k < 0 or k > n:
        return 0
        
    if k == 0 or k == n:
        return 1
        
    result = 1
    for i in range(1, k + 1):
        result *= (n - i + 1)
        result //= i
        
    return result
