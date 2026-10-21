/**
 * `window.__mrfseg`, for the browser console and the browser tests: alignment is debugged with
 * numbers, not by eye. Everything it exposes is plain data or an imperative command.
 */

import { getImage } from "@/lib/scene/dataRegistry";
import { useSceneStore } from "@/lib/scene/store";
import { describeViewports, projectToPane, resetCamera, setViewDirection } from "@/lib/viewer/activeViewport";
import { openImage } from "./openImage";

export function installDebugHook(): void {
  Object.assign(window, {
    __mrfseg: {
      describe: () => {
        const state = useSceneStore.getState();
        return {
          world: { bounds: state.worldBounds, slicePlanes: state.slicePlanes },
          datasets: Object.values(state.datasets),
          nodes: Object.values(state.nodes),
          viewports: describeViewports(),
        };
      },
      open: openImage,
      setView: setViewDirection,
      resetCamera,
      projectToPane,
      store: useSceneStore,
      getImage,
    },
  });
}
