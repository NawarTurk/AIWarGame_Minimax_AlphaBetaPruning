from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 5.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True 
    max_turns : int | None = 100
    randomize_moves : bool = True
    broker : str | None = None

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai : bool = True
    _defender_has_ai : bool = True
    e0_counter : int = 0
    level_counter: int = 0
    my_dict: dict = field(default_factory=lambda: {i: 0 for i in range(1, 10)})
    my_dict_percent: dict = field(default_factory=lambda: {i: 0 for i in range(1, 10)})


    

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    # to do Nawar, make sure each unit moves as described 
    def is_valid_move(self, coords : CoordPair) -> Tuple[bool,str]:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""

        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            # print("Invalid Coordinates!")
            return (False, "invalid")
        
        src_unit = self.get(coords.src)
        if src_unit is None or src_unit.player != self.next_player:
            # print("Please choose one of your units")
            return (False, "invalid")
        
        # dst_unit = self.get(coords.dst)
        # if (dst_unit is not None):
        #     return False
       

        UP_MOVE = "up"    
        DOWN_MOVE = "down"
        LEFT_MOVE = "left"
        RIGHT_MOVE = "right"
        IN_PLACE_MOVE = "in place"
        src_unit_type = self.get(coords.src).type
        src_player = self.get(coords.src).player
        src_unit = self.get(coords.src)
        dst_unit = self.get(coords.dst)



        move = ""

        if coords.dst.row + 1 == coords.src.row and coords.dst.col == coords.src.col:
            move = UP_MOVE
        elif coords.dst.row - 1 == coords.src.row and coords.dst.col == coords.src.col:
            move = DOWN_MOVE
        elif coords.dst.col + 1 == coords.src.col and coords.dst.row == coords.src.row: 
            move = LEFT_MOVE
        elif coords.dst.col - 1 == coords.src.col and coords.dst.row == coords.src.row:
            move = RIGHT_MOVE
        elif coords.dst.col  == coords.src.col and coords.dst.row == coords.src.row:
          move = IN_PLACE_MOVE 
        else:
            # print("Invalid move. You cannot move diagonally or to a non-adjacent block")
            return (False, "invalid")



        if dst_unit is not None:
            if (move == IN_PLACE_MOVE):
                return (True, "self-destruct")

            if src_unit.player != dst_unit.player:
                return (True, "attack")
            
            if src_unit.player == dst_unit.player and dst_unit.health < 9:
                return (True, "repair")
        

        
        if src_unit_type in (UnitType.AI, UnitType.Program, UnitType.Firewall):
            adjacent_units = []

            if coords.src.row > 0:
                adjacent_units.append(self.board[coords.src.row - 1][coords.src.col])  # above
            if coords.src.row < self.options.dim-1:  
                adjacent_units.append(self.board[coords.src.row + 1][coords.src.col])  # below
            if coords.src.col > 0:
                adjacent_units.append(self.board[coords.src.row][coords.src.col - 1])  # left
            if coords.src.col < self.options.dim-1:  
                adjacent_units.append(self.board[coords.src.row][coords.src.col + 1])  # right
            
            for unit in adjacent_units:
                if unit != None:
                    if unit.player == self.next_player.next():
                        # print("The {} unit is engaged and cannot move".format(src_unit_type.name))
                        return (False, "invalid")         
        
        
        if src_player == Player.Attacker and src_unit_type in (UnitType.AI, UnitType.Program, UnitType.Firewall) and move in (UP_MOVE, LEFT_MOVE) and self.get(coords.dst) is None:
            # print("The attacker’s {} unit can only move up or left".format(src_unit_type.name))
            return (True, "move")
        
        if src_player == Player.Defender and src_unit_type in (UnitType.AI, UnitType.Program, UnitType.Firewall) and move in (DOWN_MOVE, RIGHT_MOVE) and self.get(coords.dst) is None:
            # print("The defender’s {} unit can only move down or right".format(src_unit_type.name))
            return (True, "move")
        
        if src_unit_type in (UnitType.Tech, UnitType.Virus) and move in (UP_MOVE, DOWN_MOVE, RIGHT_MOVE, LEFT_MOVE) and self.get(coords.dst) is None:
            return (True, "move")


        return (False, "invalid")


    # Nawar Code_End

    # to do by Erwin, what does the string represetns
    def perform_move(self, coords : CoordPair) -> Tuple[bool,str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""

        # checking if it is a valid move. if it is,move it and exit the function
        
        bool, msg = self.is_valid_move(coords)

        src_unit = self.get(coords.src)
        dst_unit = self.get(coords.dst)
        
        if (msg == "move"): 
            self.set(coords.dst,self.get(coords.src))
            self.set(coords.src,None)
            return (True,"Valid move/displacement")
        
        elif (msg == "attack"):
            # Calculate damage to both units
            src_damage = src_unit.damage_amount(dst_unit)
            dst_damage = dst_unit.damage_amount(src_unit)
            # Apply damage to units
            self.mod_health(coords.src, -dst_damage)
            self.mod_health(coords.dst, -src_damage)

            return (True, "\nAttack successful")
        elif (msg == "repair"):
        # Calculate health change for the target unit
            repair_amount = src_unit.repair_amount(dst_unit)
            # Apply health change to target unit
            dst_unit.mod_health(repair_amount)
            return (True, "\nRepair successful")
        elif (msg == "self-destruct"):
            self.mod_health(coords.src, -100)
            # Apply self-destruct damage to surrounding units
            for adjacent_coord in coords.src.iter_adjacent():
                adjacent_unit = self.get(adjacent_coord)
                if adjacent_unit:
                    adjacent_unit.mod_health(-2)
                    if not adjacent_unit.is_alive():
                        self.set(adjacent_coord, None)

            # Calculate and add the diagonal coordinates
            diagonal_coords = [
                Coord(coords.src.row - 1, coords.src.col - 1),
                Coord(coords.src.row - 1, coords.src.col + 1),
                Coord(coords.src.row + 1, coords.src.col - 1),
                Coord(coords.src.row + 1, coords.src.col + 1),
            ]

            for diagonal_coord in diagonal_coords:
                diagonal_unit = self.get(diagonal_coord)
                if diagonal_unit:
                    diagonal_unit.mod_health(-2)
                    if not diagonal_unit.is_alive():
                        self.set(diagonal_coord, None)

            return (True, "\nSelf-destructed")
        else:
            return (False, "\nIllegal Move!")
            


 
        


       

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')
    
    # check if we understand what is this, what is a broker
    # Adam generate reports
    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                
                (success,result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: moved from {mv.src} to {mv.dst}",end='')
                    print(result)
                    self.next_turn()
                    print("Timeout time for this move: " + str(self.options.max_time) + " s") #time taken to do the move
                    return mv #returning the move done by user
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success,result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ",end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            print("max amount of turns reached")
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker    
        return Player.Defender


    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst

                valid, msg = self.is_valid_move(move);
                if valid :
                    yield move.clone()
            move.dst = src
            yield move.clone()

    ### e0 function ## ______________________________________________________________>
    def e0(self):
        if (self.next_player == Player.Attacker):
            attackerUnits = self.player_units(self.next_player)
            defenderUnits = self.player_units(self.next_player.next())
        else:
            defenderUnits = self.player_units(self.next_player)
            attackerUnits = self.player_units(self.next_player.next())

        attackerScore = 0
        defenderScore = 0
        score = 0

        for coordinates, unit in attackerUnits:
            if unit.type==UnitType.AI:
                attackerScore += 9999
            else:
                attackerScore += 3

        for coordinates, unit in defenderUnits:
            if unit.type==UnitType.AI:
                defenderScore += 9999
            else:
                defenderScore += 3

        score = attackerScore - defenderScore
#Del       # print(f"eee is calculated and returned {score}")
        return score
    
    ### e1 function ## ______________________________________________________________>
    def e1(self):
        # do not touch
        return
    
    ### e2 function ## ______________________________________________________________>
    def e2(self) -> int:
        # Initialize counters for both players
        attacker_strength = 0
        defender_strength = 0

        # Iterate through the game state to calculate unit strengths
        for row in self.board:
            for unit in row:
                if unit is not None:
                    if unit.player == Player.Attacker:
                        attacker_strength += unit.health
                    else:
                        defender_strength += unit.health

        # Define the weights for the unit strengths
        attacker_weight = 1
        defender_weight = -1

        # Calculate the evaluation score
        evaluation = (attacker_weight * attacker_strength) + (defender_weight * defender_strength)

        return evaluation
    
    ### generate_children## ______________for minimax later________________________________________________>
    def generate_children(self):
        children = []
        move_candidates = list(self.move_candidates())
        if len(move_candidates) > 0:
            for move in move_candidates:
                gameCopy = self.clone()
                gameCopy.perform_move(move)
                children.append(gameCopy)
            return children
        else:
            return None
    
    # do not touch
    def minimax (self, game, depth, maximizing, start_time, max_time_allowed):
        self.level_counter = self.options.max_depth - depth
        self.my_dict[self.level_counter] += 1
        game.next_turn()
        children = game.generate_children()

        elapsed_time = (datetime.now() - start_time).total_seconds()
        if depth == 0 or children == None or (elapsed_time >= 0.9 * max_time_allowed) or game.is_finished():
            # print(f"current leaf eo is {game.e0()}")
            # print(game)
            self.my_dict[self.level_counter] += 1
            self.e0_counter +=1

            return game.e0() # assuming the use of e0
        
        if maximizing:
            maxScore = float('-inf')
            for child in children:
                minimaxScore = self.minimax(child, depth-1, False, start_time, max_time_allowed)
                maxScore = max(maxScore, minimaxScore)
            return maxScore
        else:
            minScore = float('inf')
            for child in children:
                minimaxScore = self.minimax(child, depth-1, True, start_time, max_time_allowed)
                minScore = min(minScore, minimaxScore)
            return minScore



        return
    # do not touch
    def alpha_beta (self, game, depth, alpha, beta, maximizing, start_time, max_time_allowed):
        self.level_counter = self.options.max_depth - depth
        game.next_turn()
        children = game.generate_children()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if depth == 0 or children == None or (elapsed_time >= 0.9 * max_time_allowed) or game.is_finished():
            # print(f"current leaf eo is {game.e0()} of player")
            self.e0_counter +=1
            self.my_dict[self.level_counter] += 1


            return game.e0() # assuming the use of e0
        
        if maximizing:
            maxScore = float('-inf')
            for child in children:
                minimaxScore = self.alpha_beta(child, depth-1, alpha, beta, False, start_time, max_time_allowed)
                maxScore = max(maxScore, minimaxScore)
                alpha = max(alpha, minimaxScore)
                if beta <= alpha:
                    break
            return maxScore
        else:
            minScore = float('inf')
            for child in children:
                minimaxScore = self.alpha_beta(child, depth-1, True, alpha, beta, start_time, max_time_allowed)
                minScore = min(minScore, minimaxScore)
                beta = min(beta, minimaxScore)
                if beta <= alpha:
                    break
            return minScore


    ## renamed from random_move() __________________________________________________________________________>
    def generate_best_move(self, start_time, max_time_allowed) -> Tuple[int, CoordPair | None, float]:
        best_move = None
        best_score = None
        maximizing = None
        depth = self.options.max_depth; # check
        is_alpha_beta = self.options.alpha_beta

        if (self.next_player == Player.Attacker): # if the current player is attacker maximize
            maximizing = False # for the child minimize
            best_score = float('-inf')
        else:
            maximizing = True # for the child
            best_score = float('inf')

        move_candidates = list(self.move_candidates())
        if len(move_candidates) > 0:  
            if ((datetime.now() - start_time).total_seconds() > 0.95 * max_time_allowed):
                best_move = move_candidates[0]
            else:
                counter = 0

                for move in move_candidates:
                    gameCopy = self.clone()
                    gameCopy.perform_move(move)
                    counter +=1 

                    if (is_alpha_beta):
                        current_move_score = self.alpha_beta(gameCopy, depth-1, float('-inf'), float('inf'), maximizing, start_time, max_time_allowed)
                    else:
                        current_move_score = self.minimax(gameCopy, depth-1, maximizing, start_time, max_time_allowed)
 

                    if (self.next_player == Player.Attacker):
                        if (current_move_score>best_score):
                            best_score = current_move_score
                            best_move = move
                    else:
                        if (current_move_score<best_score):
                            best_score = current_move_score
                            best_move = move



            gameCopy = self.clone()
            gameCopy.perform_move(best_move)
            best_move_e_score = gameCopy.e0()


            return (best_move_e_score, best_move, 0)  
        else:
            return (0, None, 0)

        


    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        max_time_allowed = self.options.max_time
        (score, move, avg_depth) = self.generate_best_move(start_time, max_time_allowed)
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        print(f"Cumulative evals {self.e0_counter}")
        # print(f"Average recursive depth: {avg_depth:0.1f}")
        print(f"Evals per depth: ",end='')
        self.my_dict
        for k in sorted(self.my_dict.keys()):
            print(f"{k}:{self.my_dict[k]} ",end='')
            self.my_dict_percent[k] = self.my_dict[k]/self.e0_counter
        print()
        print(f"Evals per depth%: ",end='')
        for k in sorted(self.my_dict_percent.keys()):
            print(f"{k}: {self.my_dict_percent[k] * 100:.2f}% ", end='')
        print()



        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:

                        pass
                else:
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--max_turn', type=float, help='maximum amount of turns')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if args.max_turn is not None:
        options.max_turns = args.max_turn

    # create a new game
    game = Game(options=options)



    #storing number of turns that have passed
    num_turn = 0

    #storing game type so that we can input into trace file
    type_of_game = game_type._name_.split("Vs")
    game_mode = ""

    if (type_of_game[0] == "Attacker" or "Defender" and type_of_game[1] == "Attacker" or "Defender"):
        game_mode = "H-H" 
    elif (type_of_game[0] == "Attacker" or "Defender" and type_of_game[1] == "Comp" ):
        game_mode = "H-AI"
    elif (type_of_game[0] == "Comp" and type_of_game[1] == "Attacker" or "Defender" ):
        game_mode = "AI-H"
    else:
        game_mode = "Ai-Ai"
    
    # creating file with proper naming convention
    # filename = "gameTrace-" + str(game.options.alpha_beta).lower() + "-" + str(int(game.options.max_time)) + "-" + str(game.options.max_turns) + ".txt"
    # creating output file for the game
    # with open(filename, "w") as file:
    #     file.write("1. The game parameters: \n\tTimeout time: " + str(game.options.max_time) + " s\n\t" + 
    #                "Max number of turns: " + str(game.options.max_turns) + " turns\n\t" + 
    #                "Play Mode: " + game_mode + "\n")
    #     file.write("\n")
    #     file.write("2. Initial configuration of the board: \n\n" + str(game) + "\n")
    #     file.write("3. Each Action of the game: \n")
                
    # the main game loop
    while True:
        # with open(filename, "a") as file: #appending the completed moves to the created trace file
            # file.write("\n")
            print()
            print(game)
            # file.write(str(game))
            
            winner = game.has_winner()
            if winner is not None:
                print(f"{winner.name} wins in {num_turn} turns!")
                # file.write(f"\n{winner.name} wins in {num_turn} turns!")
                num_turn = num_turn + 1
                break
            if game.options.game_type == GameType.AttackerVsDefender:
                user_move = str(game.human_turn()).split(" ")                
            elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
                game.human_turn()               
            elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
                game.human_turn()                
            else:
                player = game.next_player
                move = game.computer_turn()
                if move is not None:
                    game.post_move_to_broker(move)
                else:
                    print("Computer doesn't know what to do!!!")
                    # file.write("Computer doesn't know what to do!!!")
                    exit(1)
            # file.write("Move taken: " + user_move[0] + " to " + user_move[1])
            # file.write("\n----------------------------")
            num_turn = num_turn + 1
##############################################################################################################

if __name__ == '__main__':
    main()