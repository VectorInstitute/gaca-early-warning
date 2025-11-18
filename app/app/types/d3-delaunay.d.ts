declare module 'd3-delaunay' {
  export class Delaunay {
    static from(points: number[][]): Delaunay;
    voronoi(bounds?: [number, number, number, number]): Voronoi;
  }

  export class Voronoi {
    cellPolygon(i: number): number[][] | null;
  }
}
