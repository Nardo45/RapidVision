def center_pos(dimensions, pos, return_amount=1):
    pos = [
        pos[0] - dimensions[0]//2, 
        pos[1] - dimensions[1]//2
        ]
    
    if return_amount == 1:
        return pos
    else:
        return pos[0], pos[1]