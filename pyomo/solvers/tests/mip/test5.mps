*NAME:         bell5
*ROWS:         91
*COLUMNS:      104
*INTEGER:      58
*NONZERO:      266
*BEST SOLN:    8966406.49 (opt)
*LP SOLN:      8608417.95
*SOURCE:       William Cook (Bellcore)
*              William Cook (Bellcore)
*              William Cook (Bellcore)
*APPLICATION:  fiber optic network design
*COMMENTS:     30 of the integer variables are binary
*              hard problem
*              solved with new code based on Lovasz-Scarf basis reduction
NAME          BELL5
ROWS
 N  OBJ     
 L  A1      
 L  A2      
 L  A3      
 L  A4      
 L  A5      
 L  A6      
 L  A7      
 L  A8      
 L  A9      
 L  A10     
 L  A11     
 L  A12     
 L  A13     
 L  A14     
 L  A15     
 L  B1      
 L  B2      
 L  B3      
 L  B4      
 L  B5      
 L  B6      
 L  B7      
 L  B9      
 L  B10     
 L  B11     
 L  B12     
 L  B13     
 L  B14     
 L  B16     
 L  C1      
 L  C2      
 L  C3      
 L  C4      
 L  C5      
 L  C6      
 L  C7      
 L  C8      
 L  C9      
 L  C10     
 L  C11     
 L  C12     
 L  C13     
 L  C14     
 L  C15     
 L  C16     
 L  D1      
 L  D2      
 L  D3      
 L  D4      
 L  D5      
 L  D6      
 L  D7      
 L  D9      
 L  D10     
 L  D11     
 L  D12     
 L  D13     
 L  D14     
 L  D16     
 L  E1      
 L  E2      
 L  E3      
 L  E4      
 L  E5      
 L  E6      
 L  E7      
 L  E8      
 L  E9      
 L  E10     
 L  E11     
 L  E12     
 L  E13     
 L  E14     
 L  E15     
 L  E16     
 L  F0      
 L  F1      
 L  F2      
 L  F3      
 L  F4      
 L  F5      
 L  F6      
 L  F7      
 L  F8      
 L  F9      
 L  F10     
 L  F11     
 L  F12     
 L  F13     
 L  F14     
 L  F15     
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    c1        A1                  -1   A15                 -1
    c1        B1                 -20   F0                   1
    c2        A1                   1   A2                  -1
    c2        A12                 -1   B2                 -20
    c2        F1                   1
    c3        A2                   1   A3                  -1
    c3        B3                 -20   F2                   1
    c4        A3                   1   A4                  -1
    c4        A9                  -1   B4                 -20
    c4        F3                   1
    c5        A4                   1   A5                  -1
    c5        A6                  -1   B5                 -20
    c5        F4                   1
    c6        A5                   1   B6                 -20
    c6        F5                   1
    c7        A6                   1   A7                  -1
    c7        B7                 -20   F6                   1
    c8        A7                   1   A8                  -1
    c8        F7                   1
    c9        A8                   1   B9                 -20
    c9        F8                   1
    c10       A9                   1   A10                 -1
    c10       B10                -20   F9                   1
    c11       A10                  1   A11                 -1
    c11       B11                -20   F10                  1
    c12       A11                  1   B12                -20
    c12       F11                  1
    c13       A12                  1   A13                 -1
    c13       B13                -20   F12                  1
    c14       A13                  1   A14                 -1
    c14       B14                -20   F13                  1
    c15       A14                  1   F14                  1
    c16       A15                  1   B16                -20
    c16       F15                  1
    d1        OBJ              43000   B1                   1
    d1        C1                -672
    d2        OBJ              43000   B2                   1
    d2        C2                -672
    d3        OBJ              43000   B3                   1
    d3        C3                -672
    d4        OBJ              43000   B4                   1
    d4        C4                -672
    d5        OBJ              43000   B5                   1
    d5        C5                -672
    d6        OBJ              43000   B6                   1
    d6        C6                -672
    d7        OBJ              43000   B7                   1
    d7        C7                -672
    d9        OBJ              43000   B9                   1
    d9        C9                -672
    d10       OBJ              43000   B10                  1
    d10       C10               -672
    d11       OBJ              43000   B11                  1
    d11       C11               -672
    d12       OBJ              43000   B12                  1
    d12       C12               -672
    d13       OBJ              43000   B13                  1
    d13       C13               -672
    d14       OBJ              43000   B14                  1
    d14       C14               -672
    d16       OBJ              43000   B16                  1
    d16       C16               -672
    h1        OBJ              58000   B1                   1
    h1        C1               -1344
    h2        OBJ              58000   B2                   1
    h2        C2               -1344
    h3        OBJ              58000   B3                   1
    h3        C3               -1344
    h4        OBJ              59000   B4                   1
    h4        C4               -1344
    h5        OBJ              59000   B5                   1
    h5        C5               -1344
    h6        OBJ              59000   B6                   1
    h6        C6               -1344
    h7        OBJ              59000   B7                   1
    h7        C7               -1344
    h9        OBJ              60000   B9                   1
    h9        C9               -1344
    h10       OBJ              59000   B10                  1
    h10       C10              -1344
    h11       OBJ              59000   B11                  1
    h11       C11              -1344
    h12       OBJ              59000   B12                  1
    h12       C12              -1344
    h13       OBJ              58000   B13                  1
    h13       C13              -1344
    h14       OBJ              58000   B14                  1
    h14       C14              -1344
    h16       OBJ              58000   B16                  1
    h16       C16              -1344
    g1        OBJ              10000   D1                 -24
    g2        OBJ              10000   D2                 -24
    g3        OBJ              10000   D3                 -24
    g4        OBJ              10000   D4                 -24
    g5        OBJ              10000   D5                 -24
    g6        OBJ              10000   D6                 -24
    g7        OBJ              10000   D7                 -24
    g9        OBJ              10000   D9                 -24
    g10       OBJ              10000   D10                -24
    g11       OBJ              10000   D11                -24
    g12       OBJ              10000   D12                -24
    g13       OBJ              10000   D13                -24
    g14       OBJ              10000   D14                -24
    g16       OBJ              10000   D16                -24
    MARK0001  'MARKER'                 'INTEND'
    a1        OBJ            24.5645   E1                  -1
    a1        F0             8.33E-4
    a2        OBJ            20.3962   E1                   1
    a2        E2                  -1   F1             8.33E-4
    a3        OBJ            14.1693   E2                   1
    a3        E3                  -1   F2             8.33E-4
    a4        OBJ            50.2605   E3                   1
    a4        E4                  -1   F3             8.33E-4
    a5        OBJ            58.0423   E4                   1
    a5        E5                  -1   F4             8.33E-4
    a6        OBJ            36.6095   E5                   1
    a6        E6                  -1   F5             8.33E-4
    a7        OBJ             39.201   E5                   1
    a7        E7                  -1   F6             8.33E-4
    a8        OBJ             48.034   E7                   1
    a8        E8                  -1   F7             8.33E-4
    a9        OBJ            29.4336   E8                   1
    a9        E9                  -1   F8             8.33E-4
    a10       OBJ            36.0182   E4                   1
    a10       E10                 -1   F9             8.33E-4
    a11       OBJ            18.7245   E10                  1
    a11       E11                 -1   F10            8.33E-4
    a12       OBJ            30.3169   E11                  1
    a12       E12                 -1   F11            8.33E-4
    a13       OBJ             5.3655   E2                   1
    a13       E13                 -1   F12            8.33E-4
    a14       OBJ              25.55   E13                  1
    a14       E14                 -1   F13            8.33E-4
    a15       OBJ            20.7977   E14                  1
    a15       E15                 -1   F14            8.33E-4
    a16       OBJ              1.825   E1                   1
    a16       E16                 -1   F15            8.33E-4
    b1        OBJ            2.45645   C1                  -1
    b1        F0              8.3E-5
    b2        OBJ            2.03962   C1                   1
    b2        C2                  -1   F1              8.3E-5
    b3        OBJ            1.41693   C2                   1
    b3        C3                  -1   F2              8.3E-5
    b4        OBJ            5.02605   C3                   1
    b4        C4                  -1   F3              8.3E-5
    b5        OBJ            5.80423   C4                   1
    b5        C5                  -1   F4              8.3E-5
    b6        OBJ            3.66095   C5                   1
    b6        C6                  -1   F5              8.3E-5
    b7        OBJ             3.9201   C5                   1
    b7        C7                  -1   F6              8.3E-5
    b8        OBJ             4.8034   C7                   1
    b8        C8                  -1   F7              8.3E-5
    b9        OBJ            2.94336   C8                   1
    b9        C9                  -1   F8              8.3E-5
    b10       OBJ            3.60182   C4                   1
    b10       C10                 -1   F9              8.3E-5
    b11       OBJ            1.87245   C10                  1
    b11       C11                 -1   F10             8.3E-5
    b12       OBJ            3.03169   C11                  1
    b12       C12                 -1   F11             8.3E-5
    b13       OBJ            0.53655   C2                   1
    b13       C13                 -1   F12             8.3E-5
    b14       OBJ              2.555   C13                  1
    b14       C14                 -1   F13             8.3E-5
    b15       OBJ            2.07977   C14                  1
    b15       C15                 -1   F14             8.3E-5
    b16       OBJ             0.1825   C1                   1
    b16       C16                 -1   F15             8.3E-5
    f1        C1                   1   D1                   1
    f1        E1                  -1
    f2        C2                   1   D2                   1
    f2        E2                  -1
    f3        C3                   1   D3                   1
    f3        E3                  -1
    f4        C4                   1   D4                   1
    f4        E4                  -1
    f5        C5                   1   D5                   1
    f5        E5                  -1
    f6        C6                   1   D6                   1
    f6        E6                  -1
    f7        C7                   1   D7                   1
    f7        E7                  -1
    f9        C9                   1   D9                   1
    f9        E9                  -1
    f10       C10                  1   D10                  1
    f10       E10                 -1
    f11       C11                  1   D11                  1
    f11       E11                 -1
    f12       C12                  1   D12                  1
    f12       E12                 -1
    f13       C13                  1   D13                  1
    f13       E13                 -1
    f14       C14                  1   D14                  1
    f14       E14                 -1
    f16       C16                  1   D16                  1
    f16       E16                 -1
RHS
    RHS       E1                -800   E2               -3100
    RHS       E3                 -50   E4                 -40
    RHS       E5                -900   E6                 -40
    RHS       E7               -7150   E8               -4800
    RHS       E9               -3000   E10               -250
    RHS       E11               -400   E12              -1000
    RHS       E13              -4000   E14                -90
    RHS       E15               -700   E16                -24
    RHS       F0                   8   F1                   8
    RHS       F2                   8   F3                   8
    RHS       F4                   1   F5                   1
    RHS       F6                  13   F7                  13
    RHS       F8                   1   F9                   1
    RHS       F10                 13   F11                 13
    RHS       F12                  8   F13                  8
    RHS       F14                  2   F15                  2
BOUNDS
 UP ONE       c1                   1
 UP ONE       c2                   1
 UP ONE       c3                   1
 UP ONE       c4                   1
 UP ONE       c5                   1
 UP ONE       c6                   1
 UP ONE       c7                   1
 UP ONE       c8                   1
 UP ONE       c9                   1
 UP ONE       c10                  1
 UP ONE       c11                  1
 UP ONE       c12                  1
 UP ONE       c13                  1
 UP ONE       c14                  1
 UP ONE       c15                  1
 UP ONE       c16                  1
 UP ONE       d1                   1
 UP ONE       d2                   1
 UP ONE       d3                   1
 UP ONE       d4                   1
 UP ONE       d5                   1
 UP ONE       d6                   1
 UP ONE       d7                   1
 UP ONE       d9                   1
 UP ONE       d10                  1
 UP ONE       d11                  1
 UP ONE       d12                  1
 UP ONE       d13                  1
 UP ONE       d14                  1
 UP ONE       d16                  1
 UP ONE       h1               10000
 UP ONE       h2               10000
 UP ONE       h3               10000
 UP ONE       h4               10000
 UP ONE       h5               10000
 UP ONE       h6               10000
 UP ONE       h7               10000
 UP ONE       h9               10000
 UP ONE       h10               1000
 UP ONE       h11               1000
 UP ONE       h12               1000
 UP ONE       h13               1000
 UP ONE       h14               1000
 UP ONE       h16               1000
 UP ONE       g1                1000
 UP ONE       g2                1000
 UP ONE       g3                1000
 UP ONE       g4                1000
 UP ONE       g5                1000
 UP ONE       g6                1000
 UP ONE       g7                1000
 UP ONE       g9                1000
 UP ONE       g10                100
 UP ONE       g11                100
 UP ONE       g12                100
 UP ONE       g13                100
 UP ONE       g14                100
 UP ONE       g16                100
ENDATA
