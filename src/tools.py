import hashlib

def get_hash(input_string):
    # Create a SHA-256 hash object
    sha256 = hashlib.sha256()

    # Encode the input string as bytes and update the hash object
    sha256.update(input_string.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hash_result = sha256.hexdigest()

    return hash_result

input_string = "Hello, World!"
hash_value = get_hash(input_string)
print("Hash:", hash_value)
