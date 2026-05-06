#!/usr/bin/env python3
"""
Rewrite all OpNode::execute implementations to the new ctx-based signature.
"""

import sys

UNIMPL_BODY = 'unimplemented!("Step 5/7: body migrated later")'


def find_body_end(text, start_brace_pos):
    """Given position of the opening '{', find the matching closing '}'."""
    depth = 0
    i = start_brace_pos
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def replace_execute_impls(content):
    """Replace all old execute impl signatures+bodies with the new form."""
    result = []
    pos = 0

    while pos < len(content):
        idx = content.find('fn execute(', pos)
        if idx == -1:
            result.append(content[pos:])
            break

        result.append(content[pos:idx])

        # Check within next ~600 chars if this is an OpNode execute impl
        chunk = content[idx:idx + 600]
        is_opnode_execute = (
            '&self,' in chunk
            and ('inputs' in chunk or '_inputs' in chunk)
            and ('weights' in chunk or '_weights' in chunk)
            and ('device' in chunk or '_device' in chunk)
            and 'WeightStore' in chunk
        )

        if not is_opnode_execute:
            result.append('fn execute(')
            pos = idx + len('fn execute(')
            continue

        # Find the closing ')' of the parameter list
        paren_start = idx + len('fn execute')
        depth = 0
        j = paren_start
        while j < len(content):
            if content[j] == '(':
                depth += 1
            elif content[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1

        # Find the '{' that opens the body (after -> Result<...>)
        body_open = -1
        k = j + 1
        while k < len(content):
            if content[k] == '{':
                body_open = k
                break
            k += 1

        if body_open == -1:
            result.append('fn execute(')
            pos = idx + len('fn execute(')
            continue

        body_close = find_body_end(content, body_open)
        if body_close == -1:
            result.append('fn execute(')
            pos = idx + len('fn execute(')
            continue

        # Determine indentation
        line_start = content.rfind('
', 0, idx)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        indent = content[line_start:idx]
        inner_indent = indent + '    '

        new_fn = (
            'fn execute(
'
            + inner_indent + '&self,
'
            + inner_indent + "_ctx: &mut ExecuteContext<'_, B>,
"
            + inner_indent + '_node_id: NodeId,
'
            + inner_indent + '_inputs: &[OutputRef],
'
            + indent + ') -> Result<()> {
'
            + inner_indent + UNIMPL_BODY + '
'
            + indent + '}'
        )
        result.append(new_fn)
        pos = body_close + 1

    return ''.join(result)


def process_file(path):
    with open(path, 'r') as f:
        original = f.read()

    updated = replace_execute_impls(original)

    if updated != original:
        with open(path, 'w') as f:
            f.write(updated)
        print('Updated: ' + path)
    else:
        print('No changes: ' + path)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        process_file(path)
