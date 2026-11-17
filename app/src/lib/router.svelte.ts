// Hash routing: #/, #/intro, #/game/<id>, #/play/<id>, #/dream/<id>, #/teach, #/library, #/settings.

export type Route =
  | { name: "home" }
  | { name: "intro" }
  | { name: "game"; id: string }
  | { name: "dream"; id: string }
  | { name: "play"; id: string }
  | { name: "teach" }
  | { name: "library" }
  | { name: "settings" };

function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  switch (parts[0]) {
    case "intro":
      return { name: "intro" };
    case "game":
      return parts[1] ? { name: "game", id: parts[1] } : { name: "home" };
    case "dream":
      return parts[1] ? { name: "dream", id: parts[1] } : { name: "home" };
    case "play":
      return parts[1] ? { name: "play", id: parts[1] } : { name: "home" };
    case "teach":
      return { name: "teach" };
    case "library":
      return { name: "library" };
    case "settings":
      return { name: "settings" };
    default:
      return { name: "home" };
  }
}

export const router = $state({ route: parse(location.hash), key: location.hash || "#/" });

// How many in-app steps we can go back, so Back never leaves the app.
let depth = 0;
let goingBack = false;

window.addEventListener("hashchange", () => {
  if (goingBack) {
    depth = Math.max(0, depth - 1);
    goingBack = false;
  } else {
    depth++;
  }
  router.route = parse(location.hash);
  router.key = location.hash || "#/";
});

export function href(path: string): string {
  return `#${path}`;
}

export function go(path: string, replace = false): void {
  const target = `#${path}`;
  if (replace) {
    history.replaceState(null, "", target);
    router.route = parse(target);
    router.key = target;
  } else {
    location.hash = path;
  }
}

export function back(fallback = "/"): void {
  if (depth > 0) {
    goingBack = true;
    history.back();
  } else {
    go(fallback, true);
  }
}
