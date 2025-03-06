#ifndef MATERIAL_TUNE_H
#define MATERIAL_TUNE_H

#include <iomanip>

#include "../move.h"
#include "../position.h"
#include "../types.h"
#include "../utils.h"

#include "pgn.h"


struct material_score {
  double num_pawns[2];
  double num_knights[2];
  double num_bishops[2];
  double num_rooks[2];
  double num_queens[2];
  
  double avg_diff;
  Result result;
  double pawn_val, knight_val, bishop_val, rook_val, queen_val;
  bool empty;
  
  material_score() { clear(); }
  
  material_score& operator=(const material_score& o) {
    for (int i=0; i<2; ++i) {
      num_pawns[i] = o.num_pawns[i]; num_knights[i] = o.num_knights[i]; num_bishops[i] = o.num_bishops[i];
      num_rooks[i] = o.num_rooks[i]; num_queens[i] = o.num_queens[i]; result = o.result;
    }
    pawn_val = knight_val = bishop_val = rook_val = queen_val = 1.0;
    empty = false;
    return *this;
  }

  inline void clear() {
    for (int i=0; i<2; ++i) {
      num_pawns[i] = 0; num_knights[i] = 0; num_bishops[i] = 0;
      num_rooks[i] = 0; num_queens[i] = 0; result = Result::pgn_none;
    }
    pawn_val = 1.0;
    knight_val = 3.0;
    bishop_val = 3.1;
    rook_val = 5.0;
    queen_val = 9.0;
    empty = true;
  }
  
  inline double score() {
    return pawn_val * pawns_diff() +
      knight_val * knights_diff() +
      bishop_val * bishops_diff() +
      rook_val * rooks_diff() +
      queen_val * queens_diff();
  }
  
  inline U64 encoding() {
    U64 pawn_bits = std::abs(pawns_diff());
    U64 pawn_sign = pawns_diff() < 0 ? 1 : 0;
    pawn_bits |= pawn_sign << 5; // 000x xxxx, where 5th bit is a +/- sign

    U64 knight_bits = std::abs(knights_diff());
    U64 knight_sign = knights_diff() < 0 ? 1 : 0;
    knight_bits |= knight_sign << 5;

    U64 bish_bits = std::abs(bishops_diff());
    U64 bish_sign = bishops_diff() < 0 ? 1 : 0;
    bish_bits |= bish_sign << 5;

    U64 rook_bits = std::abs(rooks_diff());
    U64 rook_sign = rooks_diff() < 0 ? 1 : 0;
    rook_bits |= rook_sign << 5;
    
    U64 queen_bits = std::abs(queens_diff());
    U64 queen_sign = queens_diff() < 0 ? 1 : 0;
    queen_bits |= queen_sign << 5;
    
    return U64(pawn_bits | (knight_bits << 8) | (bish_bits << 16) | (rook_bits << 24) | (queen_bits << 32));
  }
  
  inline bool is_empty() const { return empty; }
  
  inline bool has_pawns() const {
    return num_pawns[white] != 0 || num_pawns[black] != 0;
  }
  
  inline double knights_diff() const {
    return num_knights[white] - num_knights[black];
  }

  inline double bishops_diff() const {
    return num_bishops[white] - num_bishops[black];
  }

  inline double rooks_diff() const {
    return num_rooks[white] - num_rooks[black];
  }

  inline double queens_diff() const {
    return num_queens[white] - num_queens[black];
  }
  
  inline double pawns_diff() const {
    return num_pawns[white] - num_pawns[black];
  }

  inline double total_knights() const {
    return num_knights[white] + num_knights[black];
  }

  inline double total_bishops() const {
    return num_bishops[white] + num_bishops[black];
  }

    inline double total_rooks() const {
    return num_rooks[white] + num_rooks[black];
  }

    inline double total_queens() const {
    return num_queens[white] + num_queens[black];
  }
  
  inline double minors_diff() const {
    return (num_knights[white] + num_bishops[white]) -
      (num_knights[black] + num_bishops[black]);
  }

  inline double pieces_diff() const {
    return knights_diff() + bishops_diff() +
      rooks_diff() + queens_diff();
  }
  

  inline void refresh(const position& p) {
    clear();
    
    for (Color c = white; c <= black; ++c ) {

      U64 pawns = (c == white ? p.get_pieces<white, pawn>() : p.get_pieces<black, pawn>());
      Square * knights = (c == white ? p.squares_of<white, knight>() : p.squares_of<black, knight>());
      Square * bishops = (c == white ? p.squares_of<white, bishop>() : p.squares_of<black, bishop>());
      Square * rooks = ( c == white ? p.squares_of<white, rook>() : p.squares_of<black, rook>());
      Square * queens = (c == white ? p.squares_of<white, queen>() : p.squares_of<black, queen>());
      
        
      while(pawns) { ++num_pawns[c]; pop_lsb(pawns); }
    
      for (Square s = *knights; s != no_square; s = *++knights) { ++num_knights[c]; }
      
      for (Square s = *bishops; s != no_square; s = *++bishops) { ++num_bishops[c]; }
      
      for (Square s = *rooks; s != no_square; s = *++rooks) { ++num_rooks[c]; }
      
      for (Square s = *queens; s != no_square; s = *++queens) { ++num_queens[c]; }
    }
    empty = false;
  }  
};


class material_tune {
  std::vector<game> games;
  std::vector<material_score> md;
  
  void analyze();
  
 public:
  material_tune() { md.clear(); }
  
 material_tune(std::vector<game> g) : games(g) { md.clear(); analyze(); }    
  
};


void material_tune::analyze() {
  
  std::vector<double> totals;
  std::vector<double> scores;
  std::vector<double*> results;
  //std::vector<material_score> hash;
  std::vector<game> filtered;

  
  for (const auto& g : games) {

    // assume evenly matched opponents
    // filter down to games decided on material scores (?)
    if (g.rating_diff() > 50 || g.moves.size() > 30) { continue; }
    
    position p;
    std::istringstream fen(START_FEN);    
    p.setup(fen); // start position for each game
    size_t count = 0;
    material_score ms;    
    size_t qcount = 0;
    bool filter = false;
    
    for (const auto& m : g.moves) {

      if (m.type == quiet) ++qcount;
      else qcount = 0;

      
      if (qcount > 2 && count > 15) {

	ms.refresh(p);
	
	if (fabs(ms.score()) > 2) {	  
	  if (ms.score() > 2 && (g.result == Result::pgn_draw || g.result == Result::pgn_bwin)) { filter = true; }
	  else if (ms.score() < -2 && (g.result == Result::pgn_draw || g.result == Result::pgn_wwin)) { filter = true; }
	}
      }

      p.do_move(m);      
      ++count;	  
      
    } // end moves loop

    if (!filter) filtered.push_back(g);
  }

  std::cout << "final games after filter = " << filtered.size() << std::endl;
  
  for (const auto& g : filtered) {

    position p;
    std::istringstream fen(START_FEN);    
    p.setup(fen); // start position for each game
    size_t count = 0;
    material_score ms;    
    size_t qcount = 0;

    for (const auto& m : g.moves) {
      
      if (m.type == quiet) ++qcount;
      else qcount = 0;
      
      
      if (qcount > 2 && count > 15) {

	ms.refresh(p);
	
	{ 

	  bool found = false;
	  double score = ms.score();
	  size_t idx = 0;
	  
	  for (const auto& s : scores) {
	    if (s == score) {
	      found = true;

	      results[idx][(g.result == Result::pgn_wwin ? 2 : g.result == Result::pgn_bwin ? 0 : 1)]++;
	      totals[idx]++;
	      break;
	    }
	    else { ++idx; }
	  }

	  if (!found) {
	    double * tmp = new double[3] {0, 0, 0};
	    tmp[(g.result == Result::pgn_wwin ? 2 : g.result == Result::pgn_bwin ? 0 : 1)]++;
	    results.push_back(tmp);
	    scores.push_back(score);
	    totals.push_back(1);
	  }
	  
	}      
      }

      p.do_move(m);      
      ++count;	  

    } // end for loop over mvs
    
  } // end for loop over filtered games

  
  for (int i=0; i<scores.size(); ++i) {

    std::cout << "score = " << scores[i] <<
      " " << results[i][0] << " " << results[i][1] << " " << results[i][2] << " " << totals[i] << std::endl;        
  }
  
}


#endif
