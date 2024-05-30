byte_string = b'.\xef\xbf\xbdB;*\xef\xbf\xbd-*\xef\xbf\xbd'
decoded_string = byte_string.decode('utf-8', errors='replace')
print(decoded_string)