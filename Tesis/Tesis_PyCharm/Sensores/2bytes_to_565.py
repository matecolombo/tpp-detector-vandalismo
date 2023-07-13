# Bytes de entrada
bytes_input = b'11111111\r\n11111111\r\n00101100\r\n00000001\r\n00101100\r\n00000001\r\n'

# Decodificar los bytes y dividir en una lista de elementos
bytes_list = bytes_input.decode().split('\r\n')

# Eliminar elementos vacíos
bytes_list = list(filter(None, bytes_list))
print(bytes_list)
bytes_list = [bytearray.fromhex(element)[::-1].hex() for element in bytes_list]

# Unir los bytes de a dos y separarlos en grupos de 5-6-5 bits
words = []
for i in range(0, len(bytes_list), 2):
    byte1 = int(bytes_list[i], 2)
    byte2 = int(bytes_list[i+1], 2)
    word = ((byte1 << 8) | byte2) & 0xFFFF  # Unir los bytes en una palabra de 16 bits
    group1 = (word >> 11) & 0b11111  # Primeros 5 bits
    group2 = (word >> 5) & 0b111111  # Siguientes 6 bits
    group3 = word & 0b11111  # Últimos 5 bits
    #words.append((group1, group2, group3))
    words.append((round(group1/31*255), round(group2/63*255), round(group3/31*255)))

# Imprimir los resultados
for i, word in enumerate(words):
    print(f"Palabra {i+1}: {word}")
