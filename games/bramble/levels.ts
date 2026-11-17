// Bramble's five hand-made levels, as ASCII rows (top row first, one character per tile).
//
//   #  ground            B  brick block        ?  coin block (bump it from below)
//   =  one-way ledge     o  coin               e  patrolling enemy
//   M  moving platform, travels along the "-" track to its right (2 tiles wide)
//   V  lift, travels up along the "|" track above it (2 tiles wide)
//   f  falling platform (written "ff")         P  player start      F  goal flag
//
// The bottom row is y = 0 and y grows upwards. A tile at column x, row y covers [x, x+1] x [y, y+1].
// Jumps reach about 2.4 tiles up and 4 tiles across at full speed, so every required gap is at most 3.

export interface LevelDef {
  name: string;
  /** Sky gradient, top and horizon colours. */
  sky: [number, number];
  rows: string[];
}

export const LEVELS: LevelDef[] = [
  {
    name: "Meadow",
    sky: [0xb9c6ff, 0xf3e6ff],
    rows: [
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "                   o     B?B?B                 o       ooo                          ?",
      "            oooo  o o                  oo     o        ====ooo           ooo oo",
      "            ====                               B                         ===             B",
      "   P    ooo                      e           BBB    e            oooo e                 BB         F",
      "##################   ##################  ##################   ###############  #########################",
      "##################   ##################  ##################   ###############  #########################",
      "##################   ##################  ##################   ###############  #########################",
    ],
  },
  {
    name: "Hollow",
    sky: [0x9fd3ff, 0xe8f6ff],
    rows: [
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "                           ?B?                  o",
      "        ooo    ooo              oooooo         BBB                 oo           oooo   ooo",
      "        ===                                          oo     B   oo ff           ====",
      "   P                  e                    e      e        BB   ff         e       e           e       F",
      "###############   #############M-------##############   ########        ###############   ##################",
      "###############   #############        ##############   ########        ###############   ##################",
      "###############   #############        ##############   ########        ###############   ##################",
    ],
  },
  {
    name: "Brickworks",
    sky: [0xffb9a3, 0xffeedd],
    rows: [
      "",
      "",
      "",
      "",
      "                                                      oooooo",
      "                                                      ======",
      "                             oooooo                 |",
      "              ?B?B?                   oe   o        |                               B?B",
      "           o         ooo              ====== oo     |        oo oo           o            B",
      "           B                BB     BB               |        ff ff          BB           BBB ooo",
      "   P     BBB     e          BBe    BB               |    e              e   BB  e       BBBBB        F",
      "#####################   #####################   ####V  ######       ######################################",
      "#####################   #####################   ####   ######       ######################################",
      "#####################   #####################   ####   ######       ######################################",
    ],
  },
  {
    name: "Skyway",
    sky: [0xa7a6f2, 0xfde2f3],
    rows: [
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "                                       o                                                      ?",
      "              oooo       o         o       o   ooooo                             o",
      "                      o  ==           ff              ooo==       ooooooo       BB",
      "   P                  ==      e   ff      ff          ===     e             ===    ==     e      e       F",
      "#############M------        ######            M------       #####M--------            ########################",
      "#############               ######                          #####                     ########################",
      "#############               ######                          #####                     ########################",
    ],
  },
  {
    name: "Summit",
    sky: [0x8fa8e8, 0xfff1d6],
    rows: [
      "",
      "",
      "",
      "",
      "                                                                 ooooo",
      "                                                                 =====",
      "                                                              |",
      "           ?B?                                            o   |                                ooo   B?B",
      "                                    oooooo             o ff ==|           oooo          B      ===",
      "                 ooo         B                      o ff      |                        BBB",
      "   P    e              e   BBB e              e    ff         |     e              e  BBBBB e      e       F",
      "#################   ###############M--------#######           V  ########M------################################",
      "#################   ###############         #######              ########       ################################",
      "#################   ###############         #######              ########       ################################",
    ],
  },
];
