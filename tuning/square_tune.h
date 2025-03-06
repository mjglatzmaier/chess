#ifndef SQUARE_TUNE_H
#define SQUARE_TUNE_H

#include <iomanip>

#include "../move.h"
#include "../position.h"
#include "../types.h"
#include "../utils.h"

#include "pgn.h"


struct square_scores {
  double pawn_sc[64];
  double knight_sc[64];
  double bishop_sc[64];
  double rook_sc[64];
  double queen_sc[64];
  double king_sc[64];
  double total;

  void normalize() {
    for (Square s = A1; s <= H8; ++s) {
      pawn_sc[s] /= total;
      knight_sc[s] /= total;
      bishop_sc[s] /= total;
      rook_sc[s] /= total;
      queen_sc[s] /= total;
      king_sc[s] /= total;
    }

    double max = -1; double min = 2;    
    for (Square s = A2; s <= H7; ++s) {
      if (pawn_sc[s] > max) { max = pawn_sc[s]; }
      if (pawn_sc[s] < min) { min = pawn_sc[s]; }
    }

    for (Square s = A2; s <= H7; ++s) {
      double scaled = (pawn_sc[s] - min) / (max - min);
      pawn_sc[s] = scaled;
    }
    

    
    max = -1; min = 2;    
    for (Square s = A1; s <= H8; ++s) {
      if (knight_sc[s] > max) { max = knight_sc[s]; }
      if (knight_sc[s] < min) { min = knight_sc[s]; }
    }

    for (Square s = A1; s <= H8; ++s) {
      double scaled = (knight_sc[s] - min) / (max - min);
      knight_sc[s] = scaled;
    }


    max = -1; min = 2;    
    for (Square s = A1; s <= H8; ++s) {
      if (bishop_sc[s] > max) { max = bishop_sc[s]; }
      if (bishop_sc[s] < min) { min = bishop_sc[s]; }
    }
    
    for (Square s = A1; s <= H8; ++s) {
      double scaled = (bishop_sc[s] - min) / (max - min);
      bishop_sc[s] = scaled;
    }


    
    max = -1; min = 2;    
    for (Square s = A1; s <= H8; ++s) {
      if (rook_sc[s] > max) { max = rook_sc[s]; }
      if (rook_sc[s] < min) { min = rook_sc[s]; }
    }
    
    for (Square s = A1; s <= H8; ++s) {
      double scaled = (rook_sc[s] - min) / (max - min);
      rook_sc[s] = scaled;
    }

    max = -1; min = 2;    
    for (Square s = A1; s <= H8; ++s) {
      if (queen_sc[s] > max) { max = queen_sc[s]; }
      if (queen_sc[s] < min) { min = queen_sc[s]; }
    }
    
    for (Square s = A1; s <= H8; ++s) {
      double scaled = (queen_sc[s] - min) / (max - min);
      queen_sc[s] = scaled;
    }    
  }

  
  void update(const position& p) {
    U64 pawns = p.get_pieces<white, pawn>();
    Square * knights = p.squares_of<white, knight>();    
    Square * bishops = p.squares_of<white, bishop>();
    Square * rooks = p.squares_of<white, rook>();
    Square * queens = p.squares_of<white, queen>();
    Square * kings = p.squares_of<white, king>();

    while(pawns) { ++pawn_sc[pop_lsb(pawns)]; }
    
    for (Square s = *knights; s != no_square; s = *++knights) {
      ++knight_sc[s];
    }

    for (Square s = *bishops; s != no_square; s = *++bishops) {
      ++bishop_sc[s];
    }
    
    for (Square s = *rooks; s != no_square; s = *++rooks) {
      ++rook_sc[s];
    }
    
    for (Square s = *queens; s != no_square; s = *++queens) {
      ++queen_sc[s];
    }
    
    for (Square s = *kings; s != no_square; s = *++kings) {
      ++king_sc[s];
    }

    ++total;
  }


  void print(const Piece& p) {
    std::vector<double*> results { pawn_sc,
				   knight_sc,
				   bishop_sc,
				   rook_sc,
				   queen_sc,
				   king_sc
    };
    
    for (Row r = r1; r <= r8; ++r) {    
      //std::cout << " " << r + 1 << " | ";
      
      for (Col c = A; c <= H; ++c) {
	Square s = Square(8 * r + c);
	
	double v = results[p][s];
	std::cout << std::fixed
		  << std::setprecision(3)
		  << std::setw(3)
		  << " "
		  << v
		  << ", ";
      }
      std::cout << "  " << std::endl; 
    }
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
  }      
};


class square_tune {
  
  std::vector<game> games;
  square_scores scores;

  void analyze();
  
public:
  square_tune() { scores = {}; }
  
  square_tune(std::vector<game> g) : games(g) { scores = {}; analyze(); }

  
  void clear() { scores = {}; }
  
};







void square_tune::analyze() {

  for (const auto& g : games) {
    
    if (g.result == Result::pgn_draw ||
	g.result == Result::pgn_wwin) {
      
      position p;
      std::istringstream fen(START_FEN);
      p.setup(fen); // start position for each game
      size_t count = 0;
      for (const auto& m : g.moves) {	

	  if (p.to_move() == white && count > 15) {
	    scores.update(p);	    
	  }

	  p.do_move(m);
	  ++count;	  
      }
    }
    else continue;
  }

  scores.normalize();

  scores.print(pawn);
  scores.print(knight);
  scores.print(bishop);
  scores.print(rook);
  scores.print(queen);
  scores.print(king);
}

#endif
