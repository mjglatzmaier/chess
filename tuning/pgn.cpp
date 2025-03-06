#include "pgn.h"



pgn::pgn(const std::vector<std::string>& files) {
  pgn_files = std::vector<std::string>(files);
  parse_files();
}


bool pgn::parse_files() {

  size_t fcount = 0;
  for(const auto& f : pgn_files) {
    
    std::cout << "..parsing games from " << f << std::endl; 

    std::ifstream pgn_file(f.c_str(), std::fstream::in);

    
    if (!pgn_file.is_open()) {
      std::cout << "..failed to open pgn file, skipping" << std::endl;
      continue;
    }

    std::string line;
    game g;
    
    position p;
    std::istringstream fen(START_FEN);
    p.setup(fen); // start position for each game
    bool success = true;
    
    while (std::getline(pgn_file, line)) {

      if (is_empty(line) || is_header(line)) {
	if (is_elo(line)) parse_elo(g, line);
	continue;
      }
      
      if (!parse_moves(p, line, g)) {	
	std::cout << "..error parsing line: " << line << std::endl;
	std::cout << "..from file " << f << std::endl;
	success = false;
	continue;
      }      
      
      if (g.finished()) {
	if (success) games.push_back(g);
	g.clear();
	p.clear();
	std::istringstream tmp(START_FEN);
	p.setup(tmp);
	success = true;
      }
    }
    
    ++fcount;
    pgn_file.close();
        
  }
  std::cout << "..parsed " << games.size() << " chess games." << std::endl;

  return true;
}


bool pgn::parse_moves(position& p, const std::string& line, game& g) {
  std::stringstream ss(line);
  std::string token;
  bool success = true;
  bool comment = false;
  while(ss >> std::skipws >> token) {

    if (token.find('}') != std::string::npos) { comment = false; continue; }
	
    if (token.find('{') != std::string::npos) { comment = true; }    

    if (comment) continue;


    
    if (token == "1/2-1/2" || token == "1-0" ||
	token == "0-1" || token == "*") {
      
      g.result = (token == "1/2-1/2" ? Result::pgn_draw :
		  token == "1-0" ? Result::pgn_wwin :
		  Result::pgn_bwin);
      
      return true;
    }

    strip(token);
    
    if (token.find('.') != std::string::npos) {
      std::string segment;
      std::stringstream tmp(token);
      std::getline(tmp, segment, '.');
      std::getline(tmp, segment, '.');
      token = segment;
    }
    if (is_empty(token)) continue;
    

    
    Move m;
    if (token.size() <= 1 || !move_from_san(p, token, m)) {
      std::cout << "failed to parse move: " << token << std::endl;
      std::cout << " line = " << line << std::endl;
      success = false;
      continue;
    }
    else {
      g.moves.push_back(m);
      p.do_move(m);
    }          
  }

  return success;
}


bool pgn::move_from_san(position& p, std::string& s, Move& m) {
  
  Square to = Square::no_square;
  m.type = Movetype::no_type;
  int i = s.size() - 1;
  
  if (s == "O-O" || s == "O-O-O") {

    if (s == "O-O") to = (p.to_move() == white ? G1 : G8);
    else to = (p.to_move() == white ? C1 : C8);
    
    return find_move<king>(p, to, m);
  }


  m.type = U8(s[i] == 'N' ? promotion_n :
	      s[i] == 'B' ? promotion_b :
	      s[i] == 'R' ? promotion_r :
	      s[i] == 'Q' ? promotion_q : Movetype::no_type);    
  
  
  if (m.type != Movetype::no_type) {    
    i -= 1;    
    if (s[i] == '=') i -= 1;    
  }
  
  to = get_square(s, i);
  i -= 2;
  
  
  // normal pawn moves 'a4' or promotion a8=q
  if (i < 0 && to != Square::no_square) {
    return find_move<pawn>(p, to, m);
  }

  
  // skip capture char
  if (tolower(s[i]) == 'x') { i -= 1; }

  if (p.piece_on(to) && m.type != Movetype::no_type) {
    m.type = U8(m.type + 4); // convert to capture promotion
  }

  
  // row/col for moves with multiple candidates
  int row = -1; int col = -1;
  if (i > 0) {
    if (isdigit(s[i])) row = int(s[i] - '1');
    else col = int(s[i] - 'a');
    i -= 1;
  }


  Piece piece = Piece::no_piece; int count = 0; 
  for (const auto& sp : SanPiece) {
    if (s[0] == sp && count < 6) {
      piece = Piece(count);
      break;
    }
    ++count;
  }

  
  if (piece == Piece::no_piece) {
    col = int(s[0] - 'a');
  }
  
  
  return (piece == Piece::no_piece ? find_move<pawn>(p, to, m, row, col) :
	  piece == Piece::knight ? find_move<knight>(p, to, m, row, col) :
	  piece == Piece::bishop ? find_move<bishop>(p, to, m, row, col) :
	  piece == Piece::rook ? find_move<rook>(p, to, m, row, col) :
	  piece == Piece::queen ? find_move<queen>(p, to, m, row, col) :
	  find_move<king>(p, to, m, row, col)); 
}
