"""Encode and decode strings"""

from typing import Literal, Optional, List, Dict, Any, Union

import json
import re


def encode_json_strings(item: Any) -> str:
    if isinstance(item, (list, dict)):
        # convert to json string
        return f'```json\n{json.dumps(item)}\n```'
    else:
        return item


def decode_json_strings(encoded: str) -> List[Union[Any, str]]:
    # Regular expression pattern to find the JSON part
    pattern = r'```json\n(.*?)\n```'

    # Find all occurrences of the pattern in the encoded string
    matches = re.findall(pattern, encoded, re.DOTALL)

    results = []
    for match in matches:
        # Extract the JSON part from the encoded string
        json_string = match.strip()
        try:
            # Convert JSON string back to Python object
            results.append(json.loads(json_string))
        except (TypeError, ValueError) as e:
            # Handle decoding errors gracefully
            results.append(json_string)

    # If no matches are found, return the encoded string as is
    if not results:
        return [encoded]

    return results
