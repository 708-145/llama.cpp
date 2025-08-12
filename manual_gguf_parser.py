import sys
import struct

def read_string(f):
    len_bytes = f.read(8)
    if not len_bytes:
        return None
    length = struct.unpack('<Q', len_bytes)[0]
    return f.read(length).decode('utf-8')

def manual_extract_tensor_info(gguf_file):
    print(f"Manually parsing GGUF file: {gguf_file}")
    try:
        with open(gguf_file, 'rb') as f:
            # Read GGUF header
            magic = f.read(4)
            if magic != b'GGUF':
                print("Not a GGUF file.")
                return

            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]

            print(f"GGUF Version: {version}")
            print(f"Tensor Count: {tensor_count}")
            print(f"KV Metadata Count: {kv_count}")

            # Skip KV metadata
            for _ in range(kv_count):
                key = read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                if value_type == 0: # UINT8
                    f.read(1)
                elif value_type == 1: # INT8
                    f.read(1)
                elif value_type == 2: # UINT16
                    f.read(2)
                elif value_type == 3: # INT16
                    f.read(2)
                elif value_type == 4: # UINT32
                    f.read(4)
                elif value_type == 5: # INT32
                    f.read(4)
                elif value_type == 6: # FLOAT32
                    f.read(4)
                elif value_type == 7: # BOOL
                    f.read(1)
                elif value_type == 8: # STRING
                    read_string(f)
                elif value_type == 9: # ARRAY
                    array_type = struct.unpack('<I', f.read(4))[0]
                    array_len = struct.unpack('<Q', f.read(8))[0]
                    # This is a simplification, we just read the bytes
                    # A proper implementation would need to know the size of the array_type
                    # but for just skipping, this is a rough approximation that might work for some files
                    # A more robust solution would be needed for arbitrary GGUF files.
                    # For now, we assume a fixed size for simplicity to get to the tensor data.
                    # This part is fragile and might fail on some GGUF files.
                    if array_type in [0, 1, 7]: # 1 byte types
                        f.read(array_len)
                    elif array_type in [2, 3]: # 2 byte types
                        f.read(array_len * 2)
                    elif array_type in [4, 5, 6]: # 4 byte types
                        f.read(array_len * 4)
                    elif array_type in [10, 11, 12]: # 8 byte types
                        f.read(array_len * 8)
                    elif array_type == 8: # string
                        for _ in range(array_len):
                            read_string(f)
                elif value_type == 10: # UINT64
                    f.read(8)
                elif value_type == 11: # INT64
                    f.read(8)
                elif value_type == 12: # FLOAT64
                    f.read(8)

            # Read tensor info
            print("\nTensor Info:")
            for i in range(tensor_count):
                name = read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                type = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                print(f"  - Tensor {i}: {name}")
                print(f"    Shape: {shape}")
                print(f"    Type: {type}")
                print(f"    Offset: {offset}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manual_gguf_parser.py <path_to_gguf_file>")
        sys.exit(1)
    gguf_file = sys.argv[1]
    manual_extract_tensor_info(gguf_file)
