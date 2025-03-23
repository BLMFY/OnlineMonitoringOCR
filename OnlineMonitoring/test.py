s = "PAYPALISHIRING"
numRows = 3

def convert(s: str, numRows: int) -> str:
    result = ''
    k = 2*numRows-2
    n = len(s)//k
    for i in range(n+1):
        result += (s[k*i])

    if numRows >= 3:
        for j in range(numRows-2):
            for i in range(n+1):
                if k*i+j+1 < len(s):
                    result += (s[k*i+j+1])
                if k*i+k-(j+1) < len(s):
                    result += (s[k*i+k-(j+1)])
    
    for i in range(n+1):
        if k*i+numRows-1 < len(s):
            result += (s[k*i+numRows-1])

    return result

rr = convert(s, numRows=numRows)

print(rr)