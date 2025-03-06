#include <vector>
#include <random>
#include <functional>
#include <iostream>

#ifdef _MSC_VER
#include "direntw.h"
#else
#include <dirent.h>
#endif

#include "pbil.h"
#include "pgn.h"
#include "square_tune.h"
#include "material_tune.h"

#include "../bitboards.h"
#include "../magics.h"
#include "../zobrist.h"

void parse_args(int argc, char * argv[]);



int main(int argc, char * argv[]) {
  
  zobrist::load();
  
  bitboards::load();
  
  magics::load();

  
  parse_args(argc, argv);

  return EXIT_SUCCESS;
}


void parse_args(int argc, char * argv[]) {

  for (int j = 0; j < argc; ++j) {
    
    if (!strcmp(argv[j], "-pgn_dir")) {

      DIR * dir;
      struct dirent * ent;
      std::vector<std::string> pgn_files;
      if ((dir = opendir(argv[j + 1])) != NULL) {
        
        std::string sdir(argv[j + 1]);

        while ((ent = readdir(dir)) != NULL) {
          if (!strcmp(ent->d_name, ".") ||
            !strcmp(ent->d_name, "..")) {
            continue;
          }
          std::string fname = sdir + "/" + ent->d_name;
          pgn_files.push_back(fname);
        }
        
        closedir(dir);
        pgn io(pgn_files); // run analysis
        //square_tune st(io.parsed_games());
        material_tune mt(io.parsed_games());
      }
      else {
        std::cout << "..pgn directory "
          << argv[j + 1]
          << " not found, ignored." << std::endl;
      }
      
    } // end pgn dir check

    else if (!strcmp(argv[j], "-tune_exe")) {
      std::string exe = argv[j + 1];


    }
    
  } // endfor - options
  
  
}
