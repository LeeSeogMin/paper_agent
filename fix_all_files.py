# -*- coding: utf-8 -*-

"""
Script to remove null bytes from all Python files in the project
"""

import os
import fnmatch

def remove_null_bytes(directory, patterns=['*.py']):
    """Function to remove null bytes from all Python files"""
    fixed_files = []
    
    for root, dirnames, filenames in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                file_path = os.path.join(root, filename)
                
                try:
                    # Read file contents
                    with open(file_path, 'rb') as file:
                        content = file.read()
                    
                    # Check for null bytes
                    if b'\x00' in content:
                        print(f"Null bytes found: {file_path}")
                        
                        # Remove null bytes
                        content = content.replace(b'\x00', b'')
                        
                        # Save to new file
                        backup_path = file_path + '.bak'
                        os.rename(file_path, backup_path)
                        
                        with open(file_path, 'wb') as file:
                            file.write(content)
                        
                        fixed_files.append(file_path)
                        print(f"File modified: {file_path}")
                        
                except Exception as e:
                    print(f"Error occurred: {file_path} - {str(e)}")
    
    return fixed_files

if __name__ == "__main__":
    # Check all Python files in the current directory
    fixed = remove_null_bytes('.')
    
    if fixed:
        print(f"\nTotal {len(fixed)} files fixed:")
        for file in fixed:
            print(f"- {file}")
    else:
        print("\nNo files need fixing.") 