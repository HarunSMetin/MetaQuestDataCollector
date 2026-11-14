import { XRDevice, metaQuest3 } from "iwer";

// Hands clutch in tests: IWER's built-in "point" pose (index out; middle, ring and little finger curled).
const dev = new XRDevice(metaQuest3);
dev.installRuntime({ forceInstall: true });
window.__xr = dev;
