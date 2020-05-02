from typing import *
from math import inf
from random import random
from itertools import combinations

class AmbiguityError(BaseException):
    """Raised when ambiguity arises and there is no suitable return value"""
    pass

def legal(word: str) -> bool:
    allowed = '01'
    for b in word:
        if b not in allowed:
            return False
    return True

def uniform_length(*words: str) -> bool:
    for i in range(len(words) - 1):
        if len(words[i]) != len(words[i + 1]):
            return False
    return True

def weight(word: str) -> int:
    if not legal(word):
        raise ValueError(f"word '{word}' must consist of bits ('0' or '1')")
    w = 0
    for b in word:
        if b == '1':
            w += 1
    return w

def bitsum2(b1: chr, b2: chr) -> chr:
    if not legal(b1 + b2):
        raise ValueError("arguments to add must be bits ('0' or '1')")
    if b1 + b2 in ('00', '11'):
        return '0'
    return '1'

def bitdot2(b1: chr, b2: chr) -> chr:
    if not legal(b1 + b2):
        raise ValueError("arguments to dot must be bits ('0' or '1')")
    if b1 + b2 == '11':
        return '1'
    return '0'

def wordsum2(w1: str, w2: str) -> str:
    if len(w1) != len(w2):
        raise ValueError(f"words to add {w1} and {w2} must be of the same length")
    result = ''
    for c1, c2 in zip(w1, w2):
        result += bitsum2(c1, c2)
    return result

def worddot2(w1: str, w2: str) -> str:
    if len(w1) != len(w2):
        raise ValueError(f"words to dot {w1} and {w2} must be of the same length")
    result = ''
    for b1, b2 in zip(w1, w2):
        result += bitdot2(b1, b2)
    return result

def bitsumn(bits: str) -> chr:
    n: int = len(bits)
    if n == 1:
        return bits[0]
    if n == 2:
        return bitsum2(bits[0], bits[1])
    
    if weight(bits) % 2 == 0:
        return '0'
    return '1'

def bitdotn(bits: str) -> chr:
    n: int = len(bits)
    if n == 1:
        return bits[0]
    if n == 2:
        return bitdot2(bits[0], bits[1])
    
    if ''.join(bits) == '1' * n:
        return '1'
    return '0'

def wordsumn(*words: str) -> str:
    if not uniform_length(*words):
        raise ValueError(f"words to sum {words} must have the same length")
    
    n = len(words)
    if n == 1:
        return words[0]
    if n == 2:
        return wordsum2(words[0], words[1])
    
    wordscpy = [w for w in words]
    for i in range(n - 1):
        wordscpy[i + 1] = wordsum2(wordscpy[i], wordscpy[i + 1])
    return wordscpy[n - 1]

def worddotn(*words: str) -> str:
    if not uniform_length(*words):
        raise ValueError(f"words to dot {words} must have the same length")
    n = len(words)
    if n == 1:
        return words[0]
    if n == 2:
        return worddot2(words[0], words[1])
    
    wordscpy = [w for w in words]
    for i in range(n - 1):
        wordscpy[i + 1] = worddot2(wordscpy[i], wordscpy[i + 1])
    return wordscpy[n - 1]

def distance2(w1: str, w2: str) -> int:
    return weight(wordsum2(w1, w2))

def distancen(*codewords: str) -> int:
    min_dist = inf
    for w1, w2 in combinations(codewords, 2):
        d = distance2(w1, w2)
        if d < min_dist:
            min_dist = d
    return min_dist

def nearest(*word, others: str) -> str:
    bestWord = None
    bestDist = inf
    for other in others:
        d = distance2(word, other)
        if d < bestDist:
            bestDist = d
            bestWord = other
        elif d == bestDist:
            raise AmbiguityError("multiple nearest words found")
    return bestWord


def g(code: str) -> str:
    return code + bitsumn(code)

def get_codewords(f, inputSize) -> Set[str]:
    result = set()
    for i in range(2 ** inputSize):
        inbits = str(bin(i))[2:]
        inbits = '0' * (inputSize - len(inbits)) + inbits
        result.add(f(inbits))
    return result

def applyError(word: str):
    erroneous = ''
    for b in word:
        if random() < 0.1:
            b = bitsum2(b, '1')
        erroneous += b
    return erroneous

def threefold(w):
    return 3 * w

def int2word(n: int, size: int):
    b = str(bin(n))[2:]
    padding = '0' * (size - len(b))
    return padding + b

def bitproduct2(b1: chr, b2: chr):
    if not legal(b1 + b2):
        raise ValueError("arguments to add must be bits ('0' or '1')")
    if b1 + b2 == '11':
        return '1'
    return '0'

def encode(inputword: str, matrix: Tuple[str]) -> str:
    if len(matrix) != len(inputword):
        raise ValueError(f"""incompatible dimensions for multiplication 
                of  word {inputword} with matrix {matrix}""")
    inputsize: int = len(inputword)
    resultsize: int = len(matrix[0])
    for mtxwrd in matrix:
        if len(mtxwrd) != resultsize:
            raise ValueError(f"matrix {matrix} has invalid row lengths")

    result: str = ''

    # indices of zeros in the input word should be discarded
    significant_indices = set(i for i in range(inputsize) if inputword[i] == '1')
    
    # if the input word is zero, return zero
    if len(significant_indices) == 0:
        return '0' * resultsize
    
    significant_rows = tuple(row for row in matrix if matrix.index(row) in significant_indices)

    # return the sum of the leftover rows
    return wordsumn(*significant_rows)

def lindep2(w1: str, w2: str):
    if not legal(w1 + w2):
        raise ValueError(f"words '{w1}' and '{w2}' must consist of bits ('0' or '1')")
    
    if len(w1) != len(w2):
        raise ValueError(f"words '{w1}' and '{w2}' must be of the same length")

    return w1 == w2

def lindepn(*words: str):
    if not uniform_length(*words):
        raise ValueError(f"words to check linear dependancy {words} must have the same length")

    n = len(words)
    zero = '0' * len(words[0])
    if n == 1:
        w: str = words[0]
        return w == zero

    if n == 2:
        return lindep2(words[0], words[1])
    
    for r in range(2, n + 1):
        for combination in combinations(words, r):
            if wordsumn(combination) == zero:
                return True

    return False


def subspace(*words: str) -> bool:
    if not uniform_length(*words):
        raise ValueError(f"words to check subspace {words} must have the same length")
    n = len(words)
    l = len(words[0])

    zero = '0' * l
    if zero not in words:
        return False
    
    for r in range(2, n + 1):
        for combination in combinations(words, r):
            if wordsumn(combination) not in words:
                return False

    return True


def span(*words: str) -> List[str]:
    if not uniform_length(*words):
        raise ValueError(f"words {words} must have the same length")

    if lindepn(*words):
        raise ValueError(f"words {words} must be linearly independant to calculate the span")

    n = len(words)
    l = len(words[0])
    result = []

    for i in range(2**n):
        newword = '0' * l
        for j in range(n):
            if int2word(i, n)[j] == '1':
                newword = wordsum2(newword, words[j])
        result.append(newword)
    return result

def orthog_comp(*codewords: str):
    if not uniform_length(*codewords):
        raise ValueError(f"codewords {codewords} must have the same length")
    
    G: Iterable[str] = generating_mtx(*codewords)
    k = len(G)
    n = len(G[0])
    Gp = normalise(*G)
    # print('\n'.join(Gp))
    X = [row[k:] for row in Gp]
    Hp = X
    for row in identity(n - k):
        Hp.append(row)
    return Hp


def identity(n: int):
    result = []
    for i in range(n - 1, -1, -1):
        result.append(int2word(2**i, n))
    return result

def transpose(*rows: str):
    if not uniform_length(*rows):
        raise ValueError(f"rows {rows} must have the same length")

    rowcount = len(rows)
    colcount = len(rows[0])

    trows = []
    for x in range(colcount):
        trows.append(''.join([rows[y][x] for y in range(rowcount)]))
    
    return trows


def normalise(*rows: str) -> Tuple[List[str], Tuple[int]]:
    """
    Returns a tuple containing the normalised matrix as the first 
    element and the permutation steps which occurred on the columns 
    as the second element. E.g.
    normalise(['100011',
               '110001',
               '101001'])
    returns
    (['100101',
      '010101',
      '001101'], (5, 1))
    """
    if not uniform_length(*rows):
        raise ValueError(f"rows {rows} must have the same length")

    rowcount = len(rows)
    colcount = len(rows[0])

    # List the identity columns which we want to find,
    # e.g. 1000, 0100, 0010, 0001
    id_cols_to_find = identity(rowcount)

    # Get the indices of the columns which belong to the identity
    id_col_indices = []
    x = 0

    # print(f"first col to find: {id_cols_to_find[0]}")
    while x < colcount:
        # print(f"start: x={x}")
        if len(id_cols_to_find) <= 0:
            break

        column = ''.join([rows[y][x] for y in range(rowcount)])
        if column == id_cols_to_find[0]:
            id_col_indices.append(x)
            # print(f"found {id_cols_to_find[0]} at column {x}")
            # print(f"column {x} ({column}) must go to column {rowcount - len(id_cols_to_find)}")
            id_cols_to_find.pop(0)
            x = 0
            continue

        x += 1
        # print(f"x+=1")

    # Get the indices of the remaining columns which don't belong to the identity
    rest = []
    for x in range(colcount):
        if x not in id_col_indices:
            rest.append(x)
            # print(f"column {x} must go to column {rowcount + len(rest) - 1}")

    # It's easier to work with transposes. We will just transpose the result at the end.
    t_original_mtx = transpose(*rows)

    # Stuff the identity columns at the beginning of the new matrix
    t_normal_mtx = [t_original_mtx[y] for y in id_col_indices]
    
    # Stuff the rest of the columns at the end of the new matrix
    for idx in rest:
        t_normal_mtx.append(t_original_mtx[idx])

    return transpose(*t_normal_mtx)


def outer(w1: str, w2: str) -> Tuple[str]:
    result = []
    for i in range(len(w1)):
        row = ""
        for j in range(len(w2)):
            row += bitdot2(w1[i], w2[j])
        result.append(row)
    return tuple(result)


def rref(*rows: str) -> Tuple[str]:
    if not uniform_length(*rows):
        raise ValueError(f"rows {rows} must have the same length")
    result = [row for row in rows]
    
    rowcount, colcount = len(result), len(result[0])

    row = 0
    col = 0

    while row < rowcount and col < colcount:
        # find index of largest element in remainder of column j
        k = -1
        for y in range(row, rowcount):
            val = result[y][col]
            if val == '1':
                k = y
                break

        if k != -1:
            column = ''.join([result[y][col] for y in range(rowcount)])

            # swap rows
            if k != row:
                result[k], result[row] = result[row], result[k]

        # save the right hand side of the pivot row
        aijn = ''.join(result[row][col:])

        # column we're looking at
        thiscol = ''.join([result[y][col] for y in range(rowcount)])

        # avoid adding pivot row with itself
        thiscol = thiscol[:row] + '0' + thiscol[row + 1:]

        flip = outer(thiscol, aijn)

        for y in range(rowcount):
            if '1' in flip[y]:
                lhs = result[y][col:]
                replacement = wordsum2(result[y][col:], flip[y])
                result[y] = result[y][:col] + replacement + result[y][col + len(flip[y]):]

        # Check if we're done
        if is_rref(*result):
            return result

        row += 1
        col += 1

    return result


def strictly_increasing(numbers: Iterable[int]):
    for i in range(len(numbers) - 1):
        if numbers[i] >= numbers[i + 1]:
            return False
    return True


def is_ref(*rows: str) -> bool:
    if not uniform_length(*rows):
        raise ValueError(f"rows {rows} must have the same length")
    rowcount: int = len(rows)
    colcount: int = len(rows[0])
    zero: str = '0' * colcount

    # Find the first zero row and make sure all rows below it are also zero
    for y in range(rowcount - 1):
        if rows[y] == zero:
            for below in range(y + 1, rowcount):
                if rows[below] != zero:
                    return False
    
    # Make sure the indices of leading '1's in each non-zero row is strictly increasing
    leading_indices = []
    for y in range(rowcount):
        if rows[y] != zero:
            leading_indices.append(''.join(rows[y]).find('1'))
    
    return strictly_increasing(leading_indices)   


def is_rref(*rows: str) -> bool:
    if not uniform_length(*rows):
        raise ValueError(f"rows {rows} must have the same length")

    if not is_ref(rows):
        return False

    rowcount: int = len(rows)
    colcount: int = len(rows[0])
    zero: str = '0' * colcount

    # Make sure the columns containing a leading '1' have '0' everywhere else in the column
    leading_indices = []
    for y in range(rowcount):
        if rows[y] != zero:
            leading_indices.append(''.join(rows[y]).find('1'))

    for x in leading_indices:
        column = [rows[y][x] for y in range(rowcount)]
        if column.count('1') > 1:
            return False
    
    return True


def generating_mtx(*codewords: str):
    if not uniform_length(*codewords):
        raise ValueError(f"codewords {codewords} must have the same length")
    n = len(codewords[0])
    zero = '0' * n
    
    mtx = [c for c in codewords]
    mtx = rref(*mtx)
    
    # Remove zero rows
    mtx = [row for row in mtx if row != zero]

    return mtx


def basis(*codewords: str) -> Set[str]:
    return set(generating_mtx(*codewords))


def dim(*codewords: str) -> int:
    return len(basis(*codewords))

def rate(*codewords: str) -> float:
    if not uniform_length(*codewords):
        raise ValueError(f"codewords {codewords} must have the same length")

    n = len(codewords[0])
    k = dim(*codewords)
    return k / n


def printmtx(*rows: str) -> None:
    print('\n'.join(rows))
    print()

def construct_linear_code(n: int, k: int, d: int) -> List[str]:
    # Create the parity check: n=16, k=11, d=3 => H is 16x(16-11) = 16x5
    cols = n - k
    zero = '0' * cols

    H = identity(cols)

    forbidden = set()
    forbidden.add(zero)
    for r in range(1, d - 1):
        for combination in combinations(H, r):
            word = wordsumn(*combination)
            forbidden.add(word)

    while len(H) < n:
        # Check if we're trying to kid ourselves...
        if len(forbidden) >= 2**n:
            raise ValueError(f"no linear ({n}, {k}, {d})-code exists")

        # Try adding the sum of of d or more rows of H if it's not forbidden
        wordToAdd = None
        for r in range(d - 1, len(H)):
            for combination in combinations(H, r):
                word = wordsumn(*combination)
                if word not in forbidden:
                    wordToAdd = word
                    break

            if wordToAdd is not None:
                H.append(wordToAdd)
                break
            else:
                raise ValueError(f"no linear ({n}, {k}, {d})-code exists")
        
        # Update the forbidden set
        for r in range(1, d - 1):
            for combination in combinations(H, r):
                word = wordsumn(*combination)
                forbidden.add(word)

    # Obtain G from H using standard algorithm
    X = [H[i] for i in range(cols, len(H))]
    Hp = X + identity(cols)
    
    GpT = identity(k) + transpose(*X)
    Gp = transpose(*GpT)

    GT = [GpT[i] for i in range(k, len(GpT))] + identity(k)
    G = transpose(*GT)

    return span(*G)


def syndrome(w: str, H: Tuple[str]):
    n = len(w)
    if len(H) != n:
        raise ValueError(f"length of word ({len(w)}) must equal rows in parity check matrix ({len(H)})")

    s = ''
    # Iterate over the columns of H
    for i in range(len(H[0])):
        col = ''.join([row[i] for row in H])
        
        _sum = '0'
        for i in range(n):
            prod = bitproduct2(w[i], col[i])
            _sum = bitsum2(_sum, prod)

        s += _sum
    return s


def ext_golay_decode(w: str, logging=False) -> str:
    if len(w) != 24:
        raise ValueError(f"{w} must be of length 24")

    B = []
    for i in range(11):
        b_i = '11011100010'
        b_i = b_i[i:] + b_i[: i]
        B.append(b_i)

    B.append('1' * 11 + '0')

    for i in range(11):
        B[i] += '1'

    B = tuple(B)
    H = identity(12) + list(B)
    
    s = syndrome(w, H)
    if logging:
        print(f"Calculated s = wH = {pretty(s)}")

    if weight(s) <= 3:
        e = s + '0' * 12
        if logging:
            print(f"||s|| = ||{pretty(s)}|| = {weight(s)} <= 3 so e = (s | 0_12) = {pretty(e)}")
            print(f"Thus c is w + e = {pretty(w)} + {pretty(e)} = {pretty(wordsum2(w, e))}")
            print(f"Message is first 12 bits, {pretty(wordsum2(w, e)[:12])}")

        return wordsum2(w, e)[:12]

    if logging:
        print(f"||s|| = ||{pretty(s)}|| = {weight(s)} > 3 so moving on to next check")
    for j in range(12):
        if weight(wordsum2(s, B[j])) <= 2:
            e = wordsum2(s, B[j]) + '0'* j + '1' + '0' * (12 - j - 1)
            if logging:
                print(f"||s + bj|| = ||{pretty(s)} + {pretty(B[j])}|| = ||{pretty(wordsum2(s, B[j]))}|| = {weight(wordsum2(s, B[j]))} <= 2 for j = {j + 1}")
                print(f"So e = (s + bj | theta_j) = {pretty(e)}")
                print(f"Thus c is w + e = {pretty(w)} + {pretty(e)} = {pretty(wordsum2(w, e))}")
                print(f"Message is first 12 bits, {pretty(wordsum2(w, e)[:12])}")

            return wordsum2(w, e)[:12]

    if logging:
        for j in range(12):
            print(f"||s + bj|| = ||{pretty(s)} + {pretty(B[j])}|| = ||{pretty(wordsum2(s, B[j]))}|| = {weight(wordsum2(s, B[j]))} > 2 for j = {j + 1}")
        print("no luck with s, trying for s'...")
    sp = syndrome(s, B)
    if logging:
        print(f"Calculated s' = sB = {pretty(sp)}")
    if weight(sp) <= 3:
        e = '0' * 12 + sp
        if logging:
            print(f"||s'|| = ||{pretty(sp)}|| = {weight(sp)} <= 3 so e = (0_12 | s') = {pretty(e)}")
            print(f"Thus c is w + e = {pretty(w)} + {pretty(e)} = {pretty(wordsum2(w, e))}")
            print(f"Message is first 12 bits, {pretty(wordsum2(w, e)[:12])}")
        return wordsum2(w, e)[:12]
    
    if logging:
        print(f"||s'|| = ||{pretty(sp)}|| = {weight(sp)} > 3 so moving on to next check")
    for j in range(12):
        if weight(wordsum2(sp, B[j])) <= 2:
            e = '0'* j + '1' + '0' * (12 - j - 1) + wordsum2(sp, B[j])
            if logging:
                print(f"||s' + bj|| = ||{pretty(sp)} + {pretty(B[j])}|| = ||{pretty(wordsum2(sp, B[j]))}|| = {weight(wordsum2(sp, B[j]))} <= 2 for j = {j + 1}")
                print(f"So e = (theta_j | s' + bj) = {pretty(e)}")
                print(f"Thus c is w + e = {pretty(w)} + {pretty(e)} = {pretty(wordsum2(w, e))}")
                print(f"Message is first 12 bits, {pretty(wordsum2(w, e)[:12])}")

            return wordsum2(w, e)[:12]

    if logging:
        for j in range(12):
            print(f"||s' + bj|| = ||{pretty(sp)} + {pretty(B[j])}|| = ||{pretty(wordsum2(sp, B[j]))}|| = {weight(wordsum2(sp, B[j]))} > 2 for j = {j + 1}")

        print("no luck with s', out of options. request retransmission.")
    raise AmbiguityError(f"More than 3 errors ocurred, can't find suitable decoding")
    

def pretty(w: str) -> None:
    prettyword = ''
    for i in range(len(w)):
        prettyword += w[i] + ' ' * (((i + 1) % 3 == 0) and i != 0 and i != len(w) - 1)
    
    return prettyword

def golay_decode(w: str, logging=False):
    if len(w) != 23:
        raise ValueError(f"{w} must be of length 23")

    w_star = w + '0'
    if weight(w) % 2 == 0:
        w_star = w + '1'
        if logging:
            print(f"word {pretty(w)} has even weight, appending a '1' to get w* {pretty(w_star)} (weight {weight(w_star)})")
    else:
        if logging:
            print(f"word {pretty(w)} already has odd weight, appending a '0' to get w* {pretty(w_star)} (weight {weight(w_star)})")
    
    if logging:
        print(f"running extended golay decode on w* = {pretty(w_star)}...")
    c_star = ext_golay_decode(w_star, logging=logging)

    return c_star[: -1]

w = '110 010 000 000 010 000 001 11'.replace(' ', '')
print(pretty(golay_decode(w, logging=True)))