*NAME:         bell3a
*ROWS:         123
*COLUMNS:      133
*INTEGER:      71
*NONZERO:      347
*BEST SOLN:    878430.32 (opt)
*LP SOLN:      862578.64
*SOURCE:       William Cook (Bellcore)
*      	       William Cook (Bellcore)
*              William Cook (Bellcore)
*APPLICATION:  fiber optic network design
*COMMENTS:     39 of the integer variables are binary 
*              hard problem
*              solved with new code based on Lovasz-Scarf basis reduction 
NAME          BELL3A
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
 L  A16     
 L  A17     
 L  A18     
 L  A19     
 L  A20     
 L  A21     
 L  A22     
 L  B1      
 L  B2      
 L  B3      
 L  B4      
 L  B5      
 L  B6      
 L  B7      
 L  B9      
 L  B10     
 L  B12     
 L  B13     
 L  B15     
 L  B16     
 L  B17     
 L  B20     
 L  B21     
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
 L  C17     
 L  C18     
 L  C19     
 L  C20     
 L  C21     
 L  C22     
 L  C23     
 L  D1      
 L  D2      
 L  D3      
 L  D4      
 L  D5      
 L  D6      
 L  D7      
 L  D9      
 L  D10     
 L  D12     
 L  D13     
 L  D15     
 L  D16     
 L  D17     
 L  D20     
 L  D21     
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
 L  E17     
 L  E18     
 L  E19     
 L  E20     
 L  E21     
 L  E22     
 L  E23     
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
 L  F16     
 L  F17     
 L  F18     
 L  F19     
 L  F20     
 L  F21     
 L  F22     
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    c1        A1                  -1   B1                 -20
    c1        F0                   1
    c2        A1                   1   A2                  -1
    c2        B2                 -20   F1                   1
    c3        A2                   1   A3                  -1
    c3        B3                 -20   F2                   1
    c4        A3                   1   A4                  -1
    c4        B4                 -20   F3                   1
    c5        A4                   1   A5                  -1
    c5        A9                  -1   A14                 -1
    c5        B5                 -20   F4                   1
    c6        A5                   1   A6                  -1
    c6        A18                 -1   A21                 -1
    c6        A22                 -1   B6                 -20
    c6        F5                   1
    c7        A6                   1   A7                  -1
    c7        A20                 -1   B7                 -20
    c7        F6                   1
    c8        A7                   1   A8                  -1
    c8        F7                   1
    c9        A8                   1   B9                 -20
    c9        F8                   1
    c10       A9                   1   A10                 -1
    c10       A13                 -1   B10                -20
    c10       F9                   1
    c11       A10                  1   A11                 -1
    c11       F10                  1
    c12       A11                  1   A12                 -1
    c12       B12                -20   F11                  1
    c13       A12                  1   B13                -20
    c13       F12                  1
    c14       A13                  1   F13                  1
    c15       A14                  1   A15                 -1
    c15       B15                -20   F14                  1
    c16       A15                  1   A16                 -1
    c16       B16                -20   F15                  1
    c17       A16                  1   A17                 -1
    c17       B17                -20   F16                  1
    c18       A17                  1   F17                  1
    c19       A18                  1   A19                 -1
    c19       F18                  1
    c20       A19                  1   B20                -20
    c20       F19                  1
    c21       A20                  1   B21                -20
    c21       F20                  1
    c22       A21                  1   F21                  1
    c23       A22                  1   F22                  1
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
    d12       OBJ              43000   B12                  1
    d12       C12               -672
    d13       OBJ              43000   B13                  1
    d13       C13               -672
    d15       OBJ              43000   B15                  1
    d15       C15               -672
    d16       OBJ              43000   B16                  1
    d16       C16               -672
    d17       OBJ              44000   B17                  1
    d17       C17               -672
    d20       OBJ              43000   B20                  1
    d20       C20               -672
    d21       OBJ              43000   B21                  1
    d21       C21               -672
    h1        OBJ              58000   B1                   1
    h1        C1               -1344
    h2        OBJ              58000   B2                   1
    h2        C2               -1344
    h3        OBJ              58000   B3                   1
    h3        C3               -1344
    h4        OBJ              58000   B4                   1
    h4        C4               -1344
    h5        OBJ              58000   B5                   1
    h5        C5               -1344
    h6        OBJ              58000   B6                   1
    h6        C6               -1344
    h7        OBJ              59000   B7                   1
    h7        C7               -1344
    h9        OBJ              60000   B9                   1
    h9        C9               -1344
    h10       OBJ              59000   B10                  1
    h10       C10              -1344
    h12       OBJ              59000   B12                  1
    h12       C12              -1344
    h13       OBJ              59000   B13                  1
    h13       C13              -1344
    h15       OBJ              59000   B15                  1
    h15       C15              -1344
    h16       OBJ              59000   B16                  1
    h16       C16              -1344
    h17       OBJ              60000   B17                  1
    h17       C17              -1344
    h20       OBJ              59000   B20                  1
    h20       C20              -1344
    h21       OBJ              59000   B21                  1
    h21       C21              -1344
    g1        OBJ              10000   D1                 -24
    g2        OBJ              10000   D2                 -24
    g3        OBJ              10000   D3                 -24
    g4        OBJ              10000   D4                 -24
    g5        OBJ              10000   D5                 -24
    g6        OBJ              10000   D6                 -24
    g7        OBJ              10000   D7                 -24
    g9        OBJ              10000   D9                 -24
    g10       OBJ              10000   D10                -24
    g12       OBJ              10000   D12                -24
    g13       OBJ              10000   D13                -24
    g15       OBJ              10000   D15                -24
    g16       OBJ              10000   D16                -24
    g17       OBJ              10000   D17                -24
    g20       OBJ              10000   D20                -24
    g21       OBJ              10000   D21                -24
    MARK0001  'MARKER'                 'INTEND'
    a1        OBJ             12.775   E1                  -1
    a1        F0             8.33E-4
    a2        OBJ             16.425   E1                   1
    a2        E2                  -1   F1             8.33E-4
    a3        OBJ              18.25   E2                   1
    a3        E3                  -1   F2             8.33E-4
    a4        OBJ              21.17   E3                   1
    a4        E4                  -1   F3             8.33E-4
    a5        OBJ              18.98   E4                   1
    a5        E5                  -1   F4             8.33E-4
    a6        OBJ               14.6   E5                   1
    a6        E6                  -1   F5             8.33E-4
    a7        OBJ               36.5   E6                   1
    a7        E7                  -1   F6             8.33E-4
    a8        OBJ                 73   E7                   1
    a8        E8                  -1   F7             8.33E-4
    a9        OBJ                 73   E8                   1
    a9        E9                  -1   F8             8.33E-4
    a10       OBJ              18.25   E5                   1
    a10       E10                 -1   F9             8.33E-4
    a11       OBJ              10.95   E10                  1
    a11       E11                 -1   F10            8.33E-4
    a12       OBJ               21.9   E11                  1
    a12       E12                 -1   F11            8.33E-4
    a13       OBJ               51.1   E12                  1
    a13       E13                 -1   F12            8.33E-4
    a14       OBJ             10.585   E10                  1
    a14       E14                 -1   F13            8.33E-4
    a15       OBJ            80.8475   E5                   1
    a15       E15                 -1   F14            8.33E-4
    a16       OBJ            88.5125   E15                  1
    a16       E16                 -1   F15            8.33E-4
    a17       OBJ              95.63   E16                  1
    a17       E17                 -1   F16            8.33E-4
    a18       OBJ              25.55   E17                  1
    a18       E18                 -1   F17            8.33E-4
    a19       OBJ               14.6   E6                   1
    a19       E19                 -1   F18            8.33E-4
    a20       OBJ               58.4   E19                  1
    a20       E20                 -1   F19            8.33E-4
    a21       OBJ                 73   E7                   1
    a21       E21                 -1   F20            8.33E-4
    a22       OBJ               21.9   E6                   1
    a22       E22                 -1   F21            8.33E-4
    a23       OBJ              0.073   E6                   1
    a23       E23                 -1   F22            8.33E-4
    b1        OBJ             1.2775   C1                  -1
    b1        F0              8.3E-5
    b2        OBJ             1.6425   C1                   1
    b2        C2                  -1   F1              8.3E-5
    b3        OBJ              1.825   C2                   1
    b3        C3                  -1   F2              8.3E-5
    b4        OBJ              2.117   C3                   1
    b4        C4                  -1   F3              8.3E-5
    b5        OBJ              1.898   C4                   1
    b5        C5                  -1   F4              8.3E-5
    b6        OBJ               1.46   C5                   1
    b6        C6                  -1   F5              8.3E-5
    b7        OBJ               3.65   C6                   1
    b7        C7                  -1   F6              8.3E-5
    b8        OBJ                7.3   C7                   1
    b8        C8                  -1   F7              8.3E-5
    b9        OBJ                7.3   C8                   1
    b9        C9                  -1   F8              8.3E-5
    b10       OBJ              1.825   C5                   1
    b10       C10                 -1   F9              8.3E-5
    b11       OBJ              1.095   C10                  1
    b11       C11                 -1   F10             8.3E-5
    b12       OBJ               2.19   C11                  1
    b12       C12                 -1   F11             8.3E-5
    b13       OBJ               5.11   C12                  1
    b13       C13                 -1   F12             8.3E-5
    b14       OBJ             1.0585   C10                  1
    b14       C14                 -1   F13             8.3E-5
    b15       OBJ            8.08475   C5                   1
    b15       C15                 -1   F14             8.3E-5
    b16       OBJ            8.85125   C15                  1
    b16       C16                 -1   F15             8.3E-5
    b17       OBJ              9.563   C16                  1
    b17       C17                 -1   F16             8.3E-5
    b18       OBJ              2.555   C17                  1
    b18       C18                 -1   F17             8.3E-5
    b19       OBJ               1.46   C6                   1
    b19       C19                 -1   F18             8.3E-5
    b20       OBJ               5.84   C19                  1
    b20       C20                 -1   F19             8.3E-5
    b21       OBJ                7.3   C7                   1
    b21       C21                 -1   F20             8.3E-5
    b22       OBJ               2.19   C6                   1
    b22       C22                 -1   F21             8.3E-5
    b23       OBJ             0.0073   C6                   1
    b23       C23                 -1   F22             8.3E-5
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
    f12       C12                  1   D12                  1
    f12       E12                 -1
    f13       C13                  1   D13                  1
    f13       E13                 -1
    f15       C15                  1   D15                  1
    f15       E15                 -1
    f16       C16                  1   D16                  1
    f16       E16                 -1
    f17       C17                  1   D17                  1
    f17       E17                 -1
    f20       C20                  1   D20                  1
    f20       E20                 -1
    f21       C21                  1   D21                  1
    f21       E21                 -1
RHS
    RHS       E1                -100   E2                 -50
    RHS       E3                -200   E4                 -10
    RHS       E7                -150   E8                 -10
    RHS       E9                -500   E11                -50
    RHS       E12               -160   E13               -100
    RHS       E14               -300   E15               -100
    RHS       E16               -150   E17                -10
    RHS       E18                -20   E20               -600
    RHS       E21               -200   E22               -100
    RHS       E23                -50   F0                   2
    RHS       F1                   1   F2                   2
    RHS       F3                   1   F4                   1
    RHS       F5                  13   F6                   2
    RHS       F7                   1   F8                   2
    RHS       F9                  13   F10                  1
    RHS       F11                  3   F12                  2
    RHS       F13                  2   F14                  2
    RHS       F15                  2   F16                  2
    RHS       F17                  2   F18                  1
    RHS       F19                  1   F20                  2
    RHS       F21                 13   F22                 99
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
 UP ONE       c17                  1
 UP ONE       c18                  1
 UP ONE       c19                  1
 UP ONE       c20                  1
 UP ONE       c21                  1
 UP ONE       c22                  1
 UP ONE       c23                  1
 UP ONE       d1                   1
 UP ONE       d2                   1
 UP ONE       d3                   1
 UP ONE       d4                   1
 UP ONE       d5                   1
 UP ONE       d6                   1
 UP ONE       d7                   1
 UP ONE       d9                   1
 UP ONE       d10                  1
 UP ONE       d12                  1
 UP ONE       d13                  1
 UP ONE       d15                  1
 UP ONE       d16                  1
 UP ONE       d17                  1
 UP ONE       d20                  1
 UP ONE       d21                  1
 UP ONE       h1                1000
 UP ONE       h2                1000
 UP ONE       h3                1000
 UP ONE       h4                1000
 UP ONE       h5                1000
 UP ONE       h6                1000
 UP ONE       h7                1000
 UP ONE       h9                1000
 UP ONE       h10               1000
 UP ONE       h12               1000
 UP ONE       h13               1000
 UP ONE       h15               1000
 UP ONE       h16               1000
 UP ONE       h17               1000
 UP ONE       h20               1000
 UP ONE       h21               1000
 UP ONE       g1                1000
 UP ONE       g2                1000
 UP ONE       g3                1000
 UP ONE       g4                1000
 UP ONE       g5                1000
 UP ONE       g6                1000
 UP ONE       g7                1000
 UP ONE       g9                1000
 UP ONE       g10               1000
 UP ONE       g12               1000
 UP ONE       g13               1000
 UP ONE       g15               1000
 UP ONE       g16               1000
 UP ONE       g17               1000
 UP ONE       g20               1000
 UP ONE       g21               1000
ENDATA
